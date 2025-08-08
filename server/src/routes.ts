// server/src/routes.ts
import axios from "axios";
import { Express, Request, Response } from "express";
import Anthropic from "@anthropic-ai/sdk";
import { McpManager } from "./mcp/manager.js";
import { RegistryClient } from "./registry/client.js";

import { ChatOllama } from "@langchain/ollama";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { DynamicTool } from "@langchain/core/tools";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";


// Add a helper function to sanitize input schemas for Anthropic
function sanitizeInputSchema(schema: any): any {
  if (!schema || typeof schema !== 'object') {
    return schema;
  }
  
  // Create a copy of the schema
  const sanitizedSchema = { ...schema };
  
  // Remove oneOf, allOf, anyOf at the top level
  delete sanitizedSchema.oneOf;
  delete sanitizedSchema.allOf;
  delete sanitizedSchema.anyOf;
  
  // If we removed these operators, provide a basic schema structure
  // This ensures we don't send an empty schema
  if (Object.keys(sanitizedSchema).length === 0 || 
      (schema.oneOf !== undefined || schema.allOf !== undefined || schema.anyOf !== undefined)) {
    return {
      type: "object",
      properties: {},
      description: schema.description || "Input for this tool"
    };
  }
  
  return sanitizedSchema;
}

// Cache for server ratings
const ratingsCache = new Map<string, {
  data: { average: number, count: number, score: number },
  timestamp: number
}>();
const RATINGS_CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours (increased from 30 minutes)

// Track rate limit status to avoid hitting limits repeatedly
const ratingApiState = {
  isRateLimited: false,
  rateLimitResetTime: 0,
  rateLimitBackoff: 5 * 60 * 1000, // 5 minutes initial backoff
  consecutiveErrors: 0
};

async function getWeightedRatingScore(serverId: string): Promise<{ average: number, count: number, score: number }> {
  try {
    // Check if we're currently rate limited
    if (ratingApiState.isRateLimited && Date.now() < ratingApiState.rateLimitResetTime) {
      console.log(`Rating API is rate limited, waiting until ${new Date(ratingApiState.rateLimitResetTime).toISOString()}`);
      
      // Use cached data if available
      const cached = ratingsCache.get(serverId);
      if (cached) {
        console.log(`Using cached ratings for server ${serverId} due to rate limiting`);
        return cached.data;
      }
      
      // If no cache, return default values during rate limit
      return { average: 0, count: 0, score: 0 };
    }
    
    // Check if we have cached data that's not expired
    const cached = ratingsCache.get(serverId);
    const now = Date.now();
    
    if (cached && (now - cached.timestamp) < RATINGS_CACHE_TTL) {
      console.log(`Using cached ratings for server ${serverId}`);
      return cached.data;
    }
    
    // If not in cache or expired, fetch from API
    console.log(`Fetching ratings for server ${serverId}`);
    const response = await axios.get(`https://nanda-registry.com/api/v1/servers/${serverId}/ratings`);
    
    // Reset rate limit state on successful request
    ratingApiState.isRateLimited = false;
    ratingApiState.consecutiveErrors = 0;
    
    const ratings = response.data?.data || [];

    const count = ratings.length;
    const total = ratings.reduce((sum: number, r: any) => sum + r.rating, 0);
    const average = count > 0 ? total / count : 0;
    const score = average * count;

    const result = { average, count, score };
    
    // Cache the result
    ratingsCache.set(serverId, {
      data: result,
      timestamp: now
    });

    return result;
  } catch (error) {
    console.error(`Failed to fetch ratings for server ${serverId}:`, error);
    
    // Check for rate limit error (429)
    if (axios.isAxiosError(error) && error.response?.status === 429) {
      // Set rate limit state with exponential backoff
      ratingApiState.isRateLimited = true;
      ratingApiState.consecutiveErrors++;
      
      // Increase backoff time exponentially with consecutive errors (max 1 hour)
      const backoffTime = Math.min(
        ratingApiState.rateLimitBackoff * Math.pow(2, ratingApiState.consecutiveErrors - 1),
        60 * 60 * 1000
      );
      
      ratingApiState.rateLimitResetTime = Date.now() + backoffTime;
      console.warn(`Rate limited by ratings API. Backing off for ${backoffTime/1000} seconds until ${new Date(ratingApiState.rateLimitResetTime).toISOString()}`);
    }
    
    // If we have cached data, use it even if expired
    const cached = ratingsCache.get(serverId);
    if (cached) {
      console.log(`Using expired cached ratings for server ${serverId} due to fetch error`);
      return cached.data;
    }
    
    // Default values if no cached data available
    return { average: 0, count: 0, score: 0 };
  }
}

export function setupRoutes(app: Express, mcpManager: McpManager): void {
  // Health check endpoint for deployment
  app.get("/api/healthcheck", (req: Request, res: Response) => {
    res.status(200).json({ status: "ok" });
  });

  // Session endpoint
  app.post("/api/session", (req: Request, res: Response) => {
    console.log("API: /api/session called");
    // Create a real session using sessionManager
    const sessionManager = mcpManager.getSessionManager();
    if (!sessionManager) {
      console.error("Cannot create session: SessionManager not available");
      return res.status(500).json({ error: "Session manager not available" });
    }
    
    const sessionId = sessionManager.createSession();
    console.log(`Created new session with ID: ${sessionId}`);
    res.json({ sessionId });
  });

  // API key endpoint
  app.post("/api/settings/apikey", (req: Request, res: Response) => {
    console.log("API: /api/settings/apikey called");
    const { apiKey } = req.body;

    if (!apiKey) {
      return res.status(400).json({ error: "API key is required" });
    }

    // We would store this with the session manager
    // For now, we'll just acknowledge it
    res.json({ success: true });
  });

  // Helper function to ensure session exists
  const ensureSession = (sessionId: string): string => {
    if (!sessionId) {
      console.log("No session ID provided, creating new session");
      return mcpManager.getSessionManager().createSession();
    }
    
    // Use getOrCreateSession to handle the session
    mcpManager.getSessionManager().getOrCreateSession(sessionId);
    return sessionId;
  };

  // Update the chat completion endpoint to use Llama
  app.post("/api/chat/completions", async (req: Request, res: Response) => {
    console.log("API: /api/chat/completions called");
    const { messages, tools = true, auto_proceed = true, model = "llama3.1:8b" } = req.body;
    const rawSessionId = (req.headers["x-session-id"] as string) || "";
    
    // Ensure we have a valid session
    const sessionId = ensureSession(rawSessionId);
    
    // If the session ID changed, let the client know
    if (sessionId !== rawSessionId) {
      console.log(`Using new session ID: ${sessionId} (original was: ${rawSessionId || "empty"})`);
    }

    try {
      // Initialize LangChain ChatOllama
      const llm = new ChatOllama({
        model: model, // Default to llama3.1:8b, can be overridden
        baseUrl: process.env.OLLAMA_BASE_URL || "http://10.246.47.184:10000",
        temperature: 0.7,
      });

      // Mapping ratings to natural language
      const ratingTextMap = {
        1: "terrible",
        2: "poorly rated", 
        3: "average",
        4: "good",
        5: "excellent",
      };

      // Convert messages to LangChain format
      const convertedMessages = messages.map((msg: any) => {
        const content = typeof msg.content === "string" ? msg.content : 
          Array.isArray(msg.content) ? 
            msg.content.map((c: any) => c.text || c.content || JSON.stringify(c)).join("\n") :
            JSON.stringify(msg.content);

        switch (msg.role) {
          case "user":
            return new HumanMessage(content);
          case "assistant":
            return new AIMessage(content);
          case "system":
            return new SystemMessage(content);
          default:
            return new HumanMessage(content);
        }
      });

      // Prepare LangChain tools if enabled
      let langchainTools: DynamicTool[] = [];
      let serverUsed = null;
      let intermediateResponses = [];

      if (tools) {
        try {
          const discoveredTools = await mcpManager.discoverTools(sessionId);

          langchainTools = discoveredTools.map((tool) => {
            // Use the server rating that's already included in the tool info
            const rating = tool.rating || 0;
            const ratingLabel = ratingTextMap[Math.round(rating) || 0] || "unrated";
          
            const enhancedDescription = `${tool.description || ""} 
            (This tool runs on a ${ratingLabel} server with a ${rating.toFixed(1)}/5 rating.)`;

            return new DynamicTool({
              name: tool.name,
              description: enhancedDescription,
              func: async (input: string) => {
                try {
                  console.log(`Executing tool call: ${tool.name} with input:`, input);
                  
                  // Parse input if it's a JSON string
                  let parsedInput;
                  try {
                    parsedInput = JSON.parse(input);
                  } catch {
                    // If not JSON, treat as simple string input
                    parsedInput = { query: input, input: input };
                  }
                  
                  // Execute the tool and get the server that was used
                  const result = await mcpManager.executeToolCall(
                    sessionId,
                    tool.name,
                    parsedInput
                  );
                  
                  // Capture server info if available
                  if (result.serverInfo) {
                    serverUsed = result.serverInfo;
                  }

                  console.log(`Tool result for ${tool.name}:`, JSON.stringify(result.content));
                  
                  // Convert result content to string for Llama
                  const resultText = result.content
                    .map((c: any) => {
                      if (c.type === "text") return c.text;
                      if (c.type === "resource") return `Resource: ${c.uri || c.name || JSON.stringify(c)}`;
                      return JSON.stringify(c);
                    })
                    .join("\n");
                  
                  // Add intermediate response
                  intermediateResponses.push({
                    role: "assistant",
                    content: [{ type: "text", text: `Used tool ${tool.name}: ${resultText.substring(0, 200)}...` }],
                    timestamp: new Date(),
                  });
                  
                  return resultText;
                } catch (error) {
                  console.error(`Error executing tool ${tool.name}:`, error);
                  const errorMessage = `Error executing tool: ${error.message || "Unknown error"}`;
                  
                  // Add error to intermediate responses
                  intermediateResponses.push({
                    role: "assistant", 
                    content: [{ type: "text", text: `Error with tool ${tool.name}: ${errorMessage}` }],
                    timestamp: new Date(),
                  });
                  
                  return errorMessage;
                }
              }
            });
          });

          // Sort tools by rating
          langchainTools.sort((a, b) => {
            const aRating = discoveredTools.find(t => t.name === a.name)?.rating || 0;
            const bRating = discoveredTools.find(t => t.name === b.name)?.rating || 0;
            return bRating - aRating;
          });

          // Add system message about tool ratings for Llama
          convertedMessages.unshift(new SystemMessage(
            `You are a helpful assistant with access to various tools from different servers. 
            When suggesting tools, consider their ratings to provide the most reliable options. 
            Tools with higher ratings are generally more trusted by the community.
            Use tools when they can help answer the user's question or complete their task.
            Always explain what tools you're using and why.`
          ));

        } catch (error) {
          console.error("Error discovering tools:", error);
          // Continue without tools if there's an error
        }
      }

      let finalResponse;

      if (langchainTools.length > 0 && auto_proceed) {
        // Create a simple prompt for Llama with tool usage
        const prompt = ChatPromptTemplate.fromMessages([
          ["system", `You are a helpful assistant with access to tools. 
          Available tools: {tools}
          
          Use tools when they can help answer the user's question. 
          When you use a tool, explain what you're doing and why.
          After using tools, provide a comprehensive answer based on the results.`],
          ["placeholder", "{chat_history}"],
          ["human", "{input}"],
          ["placeholder", "{agent_scratchpad}"],
        ]);

        try {
          // Create tool-calling agent for Llama
          const agent = await createToolCallingAgent({
            llm,
            tools: langchainTools,
            prompt,
          });

          // Create agent executor
          const agentExecutor = new AgentExecutor({
            agent,
            tools: langchainTools,
            verbose: true,
            returnIntermediateSteps: true,
            maxIterations: 3, // Limit iterations for Llama
          });

          // Get the last user message for the agent
          const lastUserMessage = messages
            .filter((m: any) => m.role === "user")
            .pop();
          
          const userInput = typeof lastUserMessage?.content === "string" 
            ? lastUserMessage.content 
            : Array.isArray(lastUserMessage?.content)
              ? lastUserMessage.content.map((c: any) => c.text || c.content || JSON.stringify(c)).join("\n")
              : JSON.stringify(lastUserMessage?.content) || "";

          // Execute the agent
          const agentResult = await agentExecutor.invoke({
            input: userInput,
            chat_history: convertedMessages.slice(1, -1), // Exclude system message and last user message
            tools: langchainTools.map(t => `${t.name}: ${t.description}`).join("\n"),
          });

          // Format the final response to match Anthropic's structure
          finalResponse = {
            id: `llama-${Date.now()}`,
            content: [{ type: "text", text: agentResult.output }],
            model: model,
            role: "assistant",
            stop_reason: "end_turn",
            stop_sequence: null,
            type: "message",
            usage: {
              input_tokens: 0, // Llama/Ollama doesn't provide exact token counts
              output_tokens: 0,
            }
          };

        } catch (agentError) {
          console.error("Error with agent execution, falling back to simple LLM call:", agentError);
          
          // Fallback to simple LLM call if agent fails
          const response = await llm.invoke(convertedMessages);
          
          finalResponse = {
            id: `llama-${Date.now()}`,
            content: [{ type: "text", text: response.content }],
            model: model,
            role: "assistant",
            stop_reason: "end_turn",
            stop_sequence: null,
            type: "message",
            usage: {
              input_tokens: 0,
              output_tokens: 0,
            }
          };
        }

      } else {
        // Simple LLM call without tools
        const response = await llm.invoke(convertedMessages);
        
        finalResponse = {
          id: `llama-${Date.now()}`,
          content: [{ type: "text", text: response.content }],
          model: model,
          role: "assistant",
          stop_reason: "end_turn",
          stop_sequence: null,
          type: "message",
          usage: {
            input_tokens: 0,
            output_tokens: 0,
          }
        };
      }

      // Add server info to the response
      const responseWithServerInfo = {
        ...finalResponse,
        serverInfo: serverUsed,
        requires_confirmation: !auto_proceed && langchainTools.length > 0,
        intermediateResponses: intermediateResponses,
        toolsUsed: langchainTools.length > 0
      };

      res.json(responseWithServerInfo);
    } catch (error) {
      console.error("Error creating chat completion:", error);
      res.status(500).json({
        error: error.message || "An error occurred while processing your request",
      });
    }
  });

  // Tool discovery endpoint
  app.get("/api/tools", async (req: Request, res: Response) => {
    console.log("API: /api/tools called");
    try {
      const rawSessionId = (req.headers["x-session-id"] as string) || "";
      const sessionId = ensureSession(rawSessionId);
      const tools = await mcpManager.discoverTools(sessionId);
      res.json({ tools });
    } catch (error) {
      console.error("Error discovering tools:", error);
      res.status(500).json({
        error: error.message || "An error occurred while discovering tools",
      });
    }
  });

  // Tool execution endpoint
  app.post("/api/tools/execute", async (req: Request, res: Response) => {
    console.log("API: /api/tools/execute called");
    const { toolName, args } = req.body;
    const rawSessionId = (req.headers["x-session-id"] as string) || "";
    const sessionId = ensureSession(rawSessionId);

    if (!toolName) {
      return res.status(400).json({ error: "Tool name is required" });
    }

    // Add enhanced logging
    console.log(`🛠️ Executing tool: ${toolName}`);
    console.log(`📋 Session ID: ${sessionId}`);
    console.log(`📝 Args: ${JSON.stringify(args, (key, value) => {
      // Don't log credentials in full
      if (key === "__credentials") {
        return "[CREDENTIALS REDACTED]";
      }
      return value;
    })}`);

    try {
      const result = await mcpManager.executeToolCall(
        sessionId,
        toolName,
        args || {}
      );
      
      // Get server info for the socket event
      const serverInfo = result.serverInfo || {};
      
      // Log server info
      console.log(`Server info for tool ${toolName}:`, serverInfo);
      
      // Emit a socket event with the tool execution result
      if (req.app.get('io')) {
        const io = req.app.get('io');
        console.log('Emitting tool_executed event via socket.io');
        
        const eventData = {
          toolName,
          serverId: serverInfo.id || 'unknown',
          serverName: serverInfo.name || 'Unknown Server',
          result: {
            content: result.content || [],
            isError: false
          }
        };
        
        console.log('Event data:', JSON.stringify(eventData));
        io.emit('tool_executed', eventData);
        console.log(`Socket event emitted for tool: ${toolName}`);
      } else {
        console.warn('Socket.io not available for emitting events');
      }
      
      console.log(`✅ Tool ${toolName} executed successfully`);
      res.json(result);
    } catch (error) {
      console.error(`Error executing tool ${toolName}:`, error);
      
      // Emit error event via socket
      if (req.app.get('io')) {
        const io = req.app.get('io');
        console.log('Emitting tool_executed error event via socket.io');
        
        const errorEventData = {
          toolName,
          serverId: 'unknown',
          serverName: 'Error',
          result: {
            content: [{ type: 'text', text: `Error: ${error.message || 'Unknown error'}` }],
            isError: true
          }
        };
        
        console.log('Error event data:', JSON.stringify(errorEventData));
        io.emit('tool_executed', errorEventData);
        console.log(`Socket error event emitted for tool: ${toolName}`);
      }
      
      res.status(500).json({
        error: error.message || "An error occurred while executing the tool",
      });
    }
  });

  // Get tools that require credentials
  app.get("/api/tools/credentials", (req: Request, res: Response) => {
    console.log("API: /api/tools/credentials called with headers:", JSON.stringify(req.headers));
    try {
      const rawSessionId = (req.headers["x-session-id"] as string) || "";
      const sessionId = ensureSession(rawSessionId);
      console.log(`API: /api/tools/credentials using sessionId: ${sessionId}`);
      const tools = mcpManager.getToolsWithCredentialRequirements(sessionId);
      console.log(`API: /api/tools/credentials found ${tools.length} tools requiring credentials`);
      res.json({ tools });
    } catch (error) {
      console.error("Error getting tools with credential requirements:", error);
      res.status(500).json({
        error: error.message || "An error occurred while fetching tools",
      });
    }
  });

  // Set credentials for a tool
  app.post("/api/tools/credentials", async (req: Request, res: Response) => {
    console.log("API: /api/tools/credentials POST called");
    const { toolName, serverId, credentials } = req.body;
    const rawSessionId = (req.headers["x-session-id"] as string) || "";
    const sessionId = ensureSession(rawSessionId);

    if (!toolName || !serverId || !credentials) {
      return res.status(400).json({ 
        error: "Missing required fields. toolName, serverId, and credentials are required" 
      });
    }

    try {
      const success = await mcpManager.setToolCredentials(
        sessionId,
        toolName,
        serverId,
        credentials
      );

      if (success) {
        res.json({ success: true });
      } else {
        res.status(500).json({ 
          error: "Failed to set credentials for the tool" 
        });
      }
    } catch (error) {
      console.error(`Error setting credentials for tool ${toolName}:`, error);
      res.status(500).json({
        error: error.message || "An error occurred while setting credentials",
      });
    }
  });

  // Server registration endpoint
  app.post("/api/servers", async (req: Request, res: Response) => {
    console.log("API: /api/servers POST called with body:", JSON.stringify(req.body));
    const { id, name, url, description, types, tags, verified, rating = 0 } = req.body;

    if (!id || !name || !url) {
      return res
        .status(400)
        .json({ error: "Missing required server configuration fields" });
    }

    try {
      // Try to get detailed rating, but don't fail if we can't
      let ratingInfo = { average: rating, count: 0, score: 0 };
      
      try {
        ratingInfo = await getWeightedRatingScore(id);
        console.log(`📊 Server rating summary for ${name}: avg=${ratingInfo.average}, votes=${ratingInfo.count}, score=${ratingInfo.score}`);
      } catch (ratingError) {
        console.warn(`Unable to fetch rating for ${name}, using provided rating ${rating}:`, ratingError);
      }

      // Use the server rating we just got or fall back to the provided rating
      const serverConfig = { 
        id, 
        name, 
        url, 
        description, 
        types, 
        tags, 
        verified,
        rating: ratingInfo.average || rating 
      };
      
      const success = await mcpManager.registerServer(serverConfig);

      if (success) {
        res.json({ success: true });
      } else {
        res.status(400).json({ success: false, message: "Failed to connect to server or discover tools" });
      }
    } catch (error) {
      console.error("Error registering server:", error);
      res.status(500).json({
        error:
          error.message || "An error occurred while registering the server",
      });
    }
  });

  // Get available servers endpoint
  app.get("/api/servers", (req: Request, res: Response) => {
    console.log("API: /api/servers GET called");
    const servers = mcpManager.getAvailableServers();
    res.json({ servers });
  });

  // Registry refresh endpoint
  app.post("/api/registry/refresh", async (req: Request, res: Response) => {
    console.log("API: /api/registry/refresh called");
    try {
      // Create registry client and fetch popular servers
      const registryClient = new RegistryClient();
      const servers = await registryClient.getPopularServers();
      
      console.log(`Fetched ${servers.length} popular servers from Nanda Registry`);
      
      res.json({ 
        success: true,
        servers,
        message: `Found ${servers.length} servers in the registry` 
      });
    } catch (error) {
      console.error("Error refreshing registry servers:", error);
      res.status(500).json({
        error: error.message || "An error occurred while refreshing registry servers"
      });
    }
  });

  // Registry search endpoint
  app.get("/api/registry/search", async (req: Request, res: Response) => {
    console.log("API: /api/registry/search called with query:", req.query);
    try {
      const query = req.query.q as string;
      
      if (!query) {
        return res.status(400).json({ 
          error: "Search query is required" 
        });
      }
      
      // Create registry client and search for servers
      const registryClient = new RegistryClient();
      const servers = await registryClient.searchServers(query, {
        limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
        page: req.query.page ? parseInt(req.query.page as string) : undefined,
        tags: req.query.tags as string,
        type: req.query.type as string,
        verified: req.query.verified ? req.query.verified === 'true' : undefined
      });
      
      console.log(`Found ${servers.length} servers matching query "${query}"`);
      
      res.json({ 
        success: true,
        servers,
        query,
        message: `Found ${servers.length} servers matching "${query}"` 
      });
    } catch (error) {
      console.error("Error searching registry servers:", error);
      res.status(500).json({
        error: error.message || "An error occurred while searching registry servers"
      });
    }
  });
}
