# WebSocket Streaming API Documentation

## `/ws/stream` - Real-time Chat Streaming

Complete documentation for WebSocket streaming endpoint with RAG integration, caching, and personalization.

---

## 🔗 **Endpoint Information**

**Connection URL:**
```
ws://localhost:8000/ws/stream?user_id={user_id}&course_id={course_id}
```

**HTTP Method:** WebSocket Protocol
**Protocol Version:** WebSocket v13 (RFC 6455)
**Content-Type:** JSON messages

---

## 📋 **Connection Parameters**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `user_id` | string | ✅ Yes | Unique user identifier for session management | `"user12345"` |
| `course_id` | string | ❌ No | Optional course ID for context filtering | `"course_tik_01"` |

---

## 🔄 **Complete Message Flow**

### Connection Sequence
```mermaid
sequenceDiagram
    participant C as Client
    participant S as WebSocket Server
    participant CS as ChatService
    participant RAG as RAG Service
    participant Cache as Cache Service
    participant DB as Database

    C->>S: WebSocket Connection Request
    S->>C: Connection Acknowledgment

    Note over C,S: Main Chat Loop
    C->>S: chat_request message
    S->>C: chat_started acknowledgment

    S->>CS: chat_with_database_stream()
    CS->>Cache: Check cache
    alt Cache Hit
        CS->>Cache: Get cached response
        alt Personalization Enabled
            CS->>DB: Get user context & history
            CS->>RAG: Personalize response
            CS->>C: Stream personalized chunks
        else No Personalization
            CS->>C: Send cached response
        end
    else Cache Miss
        CS->>RAG: Query knowledge base
        RAG->>CS: Stream RAG response
        CS->>C: Stream RAG chunks
        CS->>Cache: Store response
    end

    S->>C: chat_response (final with metadata)
    C->>S: ping/pong (keep-alive)
    C->>S: disconnect
```

---

## 📤 **Request Messages (Client → Server)**

### 1. Chat Request Message

**Purpose:** Submit a query for AI response

```json
{
  "type": "chat_request",
  "query": "Apa itu manajemen infrastruktur TI dan komponen utamanya?",
  "use_personalization": true,
  "chatroom_id": "room_course_tik_001"
}
```

**Field Specifications:**

| Field | Type | Required | Default | Description |
|-------|------|----------|----------|-------------|
| `type` | string | ✅ Yes | - | Must be `"chat_request"` |
| `query` | string | ✅ Yes | - | User question or prompt (1-1000 chars) |
| `use_personalization` | boolean | ❌ No | `false` | Enable response personalization with user context |
| `chatroom_id` | string | ❌ No | `null` | Room ID for conversation history (optional) |

### 2. Ping Message

**Purpose:** Keep connection alive

```json
{
  "type": "ping"
}
```

### 3. Disconnect Message

**Purpose:** Graceful connection termination

```json
{
  "type": "disconnect"
}
```

---

## 📥 **Response Messages (Server → Client)**

### 1. Connection Established

**Trigger:** Immediately after WebSocket handshake

```json
{
  "type": "connection",
  "status": "connected",
  "user_id": "user12345",
  "course_id": "course_tik_01",
  "timestamp": 1703123456.789,
  "message": "WebSocket connected. Ready for streaming chat!",
  "config": {
    "model_comprehensive": "gpt-4o-mini",
    "model_personalized": "gpt-4o-nano",
    "rag_threshold": 0.3,
    "rag_top_k": 5,
    "openai_temperature": 0.7,
    "openai_max_tokens": 1000,
    "streaming": true,
    "cache_enabled": true,
    "personalization_enabled": true
  }
}
```

**Configuration Details:**

| Config Field | Description | Value |
|-------------|-------------|--------|
| `model_comprehensive` | Main RAG model | `gpt-4o-mini` |
| `model_personalized` | Personalization model | `gpt-4o-nano` |
| `rag_threshold` | Vector similarity threshold | `0.3` |
| `rag_top_k` | Max documents retrieved | `5` |
| `openai_temperature` | AI response randomness | `0.7` |
| `streaming` | Streaming enabled | `true` |
| `cache_enabled` | Cache integration active | `true` |
| `personalization_enabled` | Personalization available | `true` |

### 2. Chat Started

**Trigger:** After receiving `chat_request`

```json
{
  "type": "chat_started",
  "query": "Apa itu manajemen infrastruktur TI dan komponen utamanya?",
  "timestamp": 1703123457.123,
  "metadata": {
    "course_id": "course_tik_01",
    "chatroom_id": "room_course_tik_001",
    "use_personalization": true,
    "service": "simple_chat_service",
    "cache_check": "initiated"
  }
}
```

### 3. Chat Chunk (Streaming)

**Trigger:** During response streaming (multiple messages)

```json
{
  "type": "chat_chunk",
  "chunk": "Manajemen infrastruktur TI adalah proses",
  "chunk_index": 1,
  "timestamp": 1703123457.234
}
```

**Additional Examples:**
```json
{
  "type": "chat_chunk",
  "chunk": "komprehensif untuk merencanakan, mengembangkan,",
  "chunk_index": 2,
  "timestamp": 1703123457.345
}
```

```json
{
  "type": "chat_chunk",
  "chunk": "dan memelihara seluruh komponen teknologi informasi.",
  "chunk_index": 3,
  "timestamp": 1703123457.456
}
```

### 4. Final Chat Response

**Trigger:** After all chunks are sent

```json
{
  "type": "chat_response",
  "response": "Manajemen infrastruktur TI adalah proses komprehensif untuk merencanakan, mengembangkan, dan memelihara seluruh komponen teknologi informasi dalam organisasi. Ini mencakup hardware, software, jaringan, dan sumber daya manusia yang mendukung operasional bisnis.",
  "sources": [
    {
      "content": "Manajemen Infrastruktur Teknologi Informasi adalah bidang yang mencakup perencanaan, implementasi, dan pemeliharaan sistem TI.",
      "metadata": {
        "material_id": "modul_infrastruktur_06",
        "course_id": "course_tik_01",
        "filename": "06. Modul Manajemen Infrastruktur TI.pdf",
        "page": "61.1",
        "score": 0.89,
        "vector_distance": 0.11,
        "chunk_type": "definition"
      }
    },
    {
      "content": "Komponen utama dalam manajemen infrastruktur TI meliputi: A. Communication Types, B. Network Devices, C. Data Centers, D. Server and Clients",
      "metadata": {
        "material_id": "modul_infrastruktur_06",
        "course_id": "course_tik_01",
        "filename": "06. Modul Manajemen Infrastruktur TI.pdf",
        "page": "63.3",
        "score": 0.84,
        "vector_distance": 0.16,
        "chunk_type": "components"
      }
    },
    {
      "content": "Dalam perencanaan infrastruktur TI, terdapat lima tahapan utama yang harus dipertimbangkan.",
      "metadata": {
        "material_id": "modul_infrastruktur_06",
        "course_id": "course_tik_01",
        "filename": "06. Modul Manajemen Infrastruktur TI.pdf",
        "page": "62.5",
        "score": 0.79,
        "vector_distance": 0.21,
        "chunk_type": "planning"
      }
    }
  ],
  "cached": false,
  "token_usage": {
    "input_tokens": 67,
    "output_tokens": 145,
    "total_tokens": 212,
    "model_pricing": {
      "input_cost_per_1k": 0.00015,
      "output_cost_per_1k": 0.0006,
      "total_cost": 0.000247
    }
  },
  "latency_ms": 2340.5,
  "timestamp": 1703123459.678,
  "metadata": {
    "source": "rag",
    "source_type": "knowledge_base",
    "response_type": "rag_response",
    "model_used": "gpt-4o-mini",
    "personalized": false,
    "course_id": "course_tik_01",
    "cost_usd": 0.000247,
    "rag_threshold_used": 0.3,
    "documents_retrieved": 5,
    "documents_used": 3
  },
  "response_source": "rag",
  "chunks_count": 15,
  "done": true
}
```

**Source Metadata Details:**

| Metadata Field | Type | Description |
|---------------|------|-------------|
| `material_id` | string | Unique material identifier |
| `course_id` | string | Course context identifier |
| `filename` | string | Source document name |
| `page` | string | Page number with decimal precision |
| `score` | float | Cosine similarity score (0-1) |
| `vector_distance` | float | Vector distance (0-1) |
| `chunk_type` | string | Content classification (optional) |

### 5. Pong Response

**Trigger:** In response to `ping` message

```json
{
  "type": "pong",
  "timestamp": 1703123500.123
}
```

### 6. Error Response

**Trigger:** Various error conditions

```json
{
  "type": "error",
  "error": "Query cannot be empty. Please provide a valid question.",
  "error_code": "EMPTY_QUERY",
  "timestamp": 1703123456.789,
  "recoverable": true,
  "suggestion": "Please resend with a valid query in the chat_request message."
}
```

**Error Types:**

| Error Code | Description | Recoverable |
|------------|-------------|--------------|
| `EMPTY_QUERY` | Query field is empty | ✅ Yes |
| `INVALID_MESSAGE_TYPE` | Unsupported message type | ✅ Yes |
| `JSON_PARSE_ERROR` | Malformed JSON | ✅ Yes |
| `SERVICE_UNAVAILABLE` | RAG/Cache service down | ❌ No |
| `RATE_LIMIT_EXCEEDED` | Too many requests | ❌ No |
| `CONNECTION_TIMEOUT` | WebSocket timeout | ❌ No |

---

## 🏗️ **Internal Service Architecture**

### Request Processing Pipeline

```mermaid
graph TD
    A[WebSocket Request] --> B[Message Validation]
    B --> C[ChatService.chat_with_database_stream]
    C --> D[CacheService.query]
    D -->{Cache Hit?}
    D -->|Yes| E[Cache Response Path]
    D -->|No| F[RAG Query Path]

    E --> G[Personalization Check]
    F --> H[UnifiedRAGService.query]
    H --> I[Knowledge Base Search]
    I --> J[Response Generation]
    J --> K[Cache Storage]

    G --> L[Stream Response]
    K --> L
    L --> M[Database Tracking]
    M --> N[WebSocket Streaming]

    style A fill:#e1f5fe
    style N fill:#c8e6c9
    style D fill:#fff3e0
    style G fill:#fce4ec
```

### Service Interactions

1. **WebSocket Layer** (`websocket_chat.py`)
   - Connection management
   - Message routing
   - Error handling

2. **SimpleChatService** (`simple_chat_service.py`)
   - Cache orchestration
   - RAG integration
   - Personalization coordination
   - Database tracking

3. **UnifiedRAGService** (`unified_rag_service.py`)
   - Knowledge base search
   - Document retrieval with page numbers
   - Response generation
   - Token tracking

4. **CacheService** (`custom_cache_service.py`)
   - Semantic search
   - Response storage
   - Similarity scoring

---

## 🔧 **Configuration & Performance**

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL_COMPREHENSIVE=gpt-4o-mini
OPENAI_MODEL_PERSONALIZED=gpt-4o-nano
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000

# RAG Configuration
RAG_DISTANCE_THRESHOLD=0.3
RAG_TOP_K=5
VECTOR_DIMENSION=1536

# Redis Configuration
REDIS_KNOWLEDGE_URL=redis://localhost:6379
REDIS_CACHE_URL=redis://localhost:6380

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
```

### Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Response Time | <3000ms | ~2340ms |
| Cache Hit Rate | >40% | Varies |
| Chunk Rate | ~100ms/chunk | ~156ms/chunk |
| Connection Time | <100ms | ~50ms |
| Memory Usage | <512MB | ~256MB |

---

## 💻 **Client Implementation Examples**

### JavaScript (Browser)

```javascript
class ChatStreamClient {
    constructor(userId, courseId = null) {
        this.userId = userId;
        this.courseId = courseId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        const url = `ws://localhost:8000/ws/stream?user_id=${this.userId}&course_id=${this.courseId}`;

        this.ws = new WebSocket(url);

        this.ws.onopen = (event) => {
            console.log('🔗 Connected to streaming chat');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onclose = (event) => {
            console.log('🔌 Connection closed');
            this.handleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
        };
    }

    handleMessage(message) {
        switch(message.type) {
            case 'connection':
                console.log('✅ Connection established:', message.config);
                this.onConnection(message);
                break;

            case 'chat_started':
                console.log('📝 Chat started for:', message.query);
                this.onChatStarted(message);
                break;

            case 'chat_chunk':
                console.log('📦 Chunk received:', message.chunk);
                this.onChunk(message);
                break;

            case 'chat_response':
                console.log('✅ Complete response received');
                console.log('📄 Sources:', message.sources.length, 'documents');
                this.onCompleteResponse(message);
                break;

            case 'error':
                console.error('❌ Error:', message.error);
                this.onError(message);
                break;

            case 'pong':
                console.log('🏓 Pong received');
                break;
        }
    }

    sendQuery(query, usePersonalization = false, chatroomId = null) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }

        const request = {
            type: 'chat_request',
            query: query.trim(),
            use_personalization: usePersonalization,
            chatroom_id: chatroomId
        };

        this.ws.send(JSON.stringify(request));
    }

    ping() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.send(JSON.stringify({ type: 'disconnect' }));
            this.ws.close();
        }
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Reconnecting... Attempt ${this.reconnectAttempts}`);

            setTimeout(() => {
                this.connect();
            }, 1000 * Math.pow(2, this.reconnectAttempts)); // Exponential backoff
        }
    }

    // Event callbacks (override in your implementation)
    onConnection(message) {}
    onChatStarted(message) {}
    onChunk(message) {}
    onCompleteResponse(message) {}
    onError(message) {}
}

// Usage Example
const client = new ChatStreamClient('student123', 'course_tik_01');

client.onConnection = (message) => {
    console.log('Connected with config:', message.config);
};

client.onChunk = (message) => {
    // Append to streaming response
    const responseDiv = document.getElementById('response');
    responseDiv.textContent += message.chunk;
};

client.onCompleteResponse = (message) => {
    console.log('Full response:', message.response);

    // Display sources with page numbers
    const sourcesDiv = document.getElementById('sources');
    message.sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.innerHTML = `
            <h4>📄 Source ${index + 1}: ${source.metadata.filename}</h4>
            <p><strong>Page:</strong> ${source.metadata.page}</p>
            <p><strong>Relevance:</strong> ${(source.metadata.score * 100).toFixed(1)}%</p>
            <p>${source.content.substring(0, 200)}...</p>
        `;
        sourcesDiv.appendChild(sourceItem);
    });

    console.log('Tokens used:', message.token_usage.total_tokens);
    console.log('Cost:', `$${message.metadata.cost_usd.toFixed(6)}`);
};

client.connect();

// Send a query
client.sendQuery(
    'Jelaskan komponen utama manajemen infrastruktur TI',
    true, // enable personalization
    'room_discussion_001'
);
```

### Python Client

```python
import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ChatMessage:
    type: str
    data: Dict[str, Any]
    timestamp: float

class ChatStreamClient:
    def __init__(self, user_id: str, course_id: Optional[str] = None):
        self.user_id = user_id
        self.course_id = course_id
        self.ws_url = f"ws://localhost:8000/ws/stream?user_id={user_id}&course_id={course_id or ''}"
        self.websocket = None
        self.is_connected = False

        # Event handlers
        self.on_connection = None
        self.on_chunk = None
        self.on_complete = None
        self.on_error = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            self.logger.info(f"🔗 Connected to {self.ws_url}")

            # Start message listener
            asyncio.create_task(self._listen_for_messages())

        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            raise

    async def _listen_for_messages(self):
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                chat_message = ChatMessage(
                    type=data.get('type', 'unknown'),
                    data=data,
                    timestamp=data.get('timestamp', 0)
                )

                await self._handle_message(chat_message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("🔌 Connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")

    async def _handle_message(self, message: ChatMessage):
        """Route message to appropriate handler"""
        self.logger.info(f"📥 Received: {message.type}")

        if message.type == "connection":
            if self.on_connection:
                await self.on_connection(message.data)

        elif message.type == "chat_started":
            self.logger.info(f"📝 Chat started: {message.data.get('query', '')}")

        elif message.type == "chat_chunk":
            if self.on_chunk:
                await self.on_chunk(message.data)

        elif message.type == "chat_response":
            self.logger.info(f"✅ Complete response received")
            if self.on_complete:
                await self.on_complete(message.data)

        elif message.type == "error":
            self.logger.error(f"❌ Error: {message.data.get('error', 'Unknown error')}")
            if self.on_error:
                await self.on_error(message.data)

        elif message.type == "pong":
            self.logger.info("🏓 Pong received")

    async def send_query(self, query: str, use_personalization: bool = False,
                       chatroom_id: Optional[str] = None):
        """Send chat request"""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")

        request = {
            "type": "chat_request",
            "query": query.strip(),
            "use_personalization": use_personalization,
            "chatroom_id": chatroom_id
        }

        await self.websocket.send(json.dumps(request))
        self.logger.info(f"📤 Sent query: {query[:50]}...")

    async def ping(self):
        """Send ping message"""
        if self.is_connected:
            await self.websocket.send(json.dumps({"type": "ping"}))

    async def disconnect(self):
        """Graceful disconnection"""
        if self.is_connected:
            await self.websocket.send(json.dumps({"type": "disconnect"}))
            await self.websocket.close()
            self.is_connected = False

# Usage Example
async def main():
    # Create client
    client = ChatStreamClient("student123", "course_tik_01")

    # Set up event handlers
    async def handle_connection(data):
        print("🔗 Connected with configuration:")
        print(f"  Model: {data['config']['model_comprehensive']}")
        print(f"  Streaming: {data['config']['streaming']}")
        print(f"  Cache: {data['config']['cache_enabled']}")

    async def handle_chunk(data):
        print(f"📦 Chunk: {data['chunk']}", end="", flush=True)

    async def handle_complete(data):
        print(f"\n\n✅ Complete response:")
        print(f"{data['response']}")

        print(f"\n📄 Sources ({len(data['sources'])} documents):")
        for i, source in enumerate(data['sources'], 1):
            metadata = source['metadata']
            print(f"  {i}. {metadata['filename']}")
            print(f"     📖 Page: {metadata['page']}")
            print(f"     🎯 Score: {metadata['score']:.3f}")
            print(f"     📝 {source['content'][:100]}...")

        print(f"\n💰 Usage:")
        print(f"  Tokens: {data['token_usage']['total_tokens']}")
        print(f"  Cost: ${data['metadata']['cost_usd']:.6f}")
        print(f"  Latency: {data['latency_ms']:.1f}ms")

    async def handle_error(data):
        print(f"❌ Error: {data['error']}")
        if data.get('recoverable', False):
            print("💡 This error is recoverable")

    # Attach handlers
    client.on_connection = handle_connection
    client.on_chunk = handle_chunk
    client.on_complete = handle_complete
    client.on_error = handle_error

    # Connect and send query
    await client.connect()

    # Send a chat request
    await client.send_query(
        query="Jelaskan pentingnya manajemen infrastruktur TI dalam organisasi modern",
        use_personalization=True,
        chatroom_id="room_discussion_001"
    )

    # Keep connection alive for demonstration
    await asyncio.sleep(10)

    # Disconnect
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔍 **Debugging & Monitoring**

### WebSocket Status Endpoint

**GET `/ws/status`**

```json
{
  "websocket_status": "active",
  "active_connections": 3,
  "connected_users": [
    "user123",
    "student456",
    "instructor789"
  ],
  "endpoint": "/ws/stream",
  "server_time": "2024-12-01T10:30:00Z",
  "uptime": 86400,
  "message_types": [
    "chat_request",
    "ping",
    "disconnect"
  ],
  "response_types": [
    "connection",
    "chat_started",
    "chat_chunk",
    "chat_response",
    "error",
    "pong"
  ],
  "supported_features": {
    "streaming": true,
    "database_integration": true,
    "cache_integration": true,
    "token_tracking": true,
    "bidirectional": true,
    "personalization": true,
    "sources_metadata": true,
    "page_tracking": true,
    "cost_tracking": true
  },
  "performance_stats": {
    "avg_response_time_ms": 2340.5,
    "cache_hit_rate": 0.42,
    "avg_tokens_per_response": 187,
    "active_chats": 3,
    "total_requests_today": 1247
  }
}
```

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
```

Debug information includes:
- Connection lifecycle events
- Message routing details
- Cache hit/miss decisions
- RAG retrieval statistics
- Token usage tracking
- Error stack traces

---

## 📋 **Troubleshooting Guide**

### Common Issues

1. **Connection Fails**
   - Check WebSocket URL parameters
   - Verify server is running on correct port
   - Check firewall/proxy settings

2. **No Response**
   - Verify `user_id` is provided
   - Check `query` field is not empty
   - Monitor server logs for errors

3. **Missing Page Numbers**
   - Ensure knowledge base documents have page metadata
   - Verify vector store schema includes `page` field
   - Check document ingestion process

4. **High Latency**
   - Monitor RAG threshold settings
   - Check knowledge base size
   - Verify Redis performance

5. **Connection Drops**
   - Implement ping/pong keep-alive
   - Add reconnection logic
   - Monitor network stability

### Error Resolution

| Error | Solution |
|--------|----------|
| `WebSocket connection failed` | Check URL, verify server status, restart client |
| `Empty query error` | Validate input before sending `chat_request` |
| `Service unavailable` | Check server logs, restart services if needed |
| `JSON parse error` | Ensure messages are valid JSON strings |
| `Rate limit exceeded` | Implement request throttling, add delays |

---

## 📈 **Performance Optimization**

### Client-Side Optimizations

1. **Chunk Buffering**
   ```javascript
   let buffer = '';
   let renderTimeout;

   function onChunk(chunk) {
       buffer += chunk.data;

       // Debounce rendering for smoother UX
       clearTimeout(renderTimeout);
       renderTimeout = setTimeout(() => {
           updateResponse(buffer);
       }, 50);
   }
   ```

2. **Connection Pooling**
   ```javascript
   class ConnectionPool {
       constructor(maxConnections = 3) {
           this.connections = [];
           this.maxConnections = maxConnections;
       }

       getConnection(userId) {
           // Reuse existing connections when possible
       }
   }
   ```

3. **Request Debouncing**
   ```javascript
   let debounceTimeout;

   function sendQueryDebounced(query) {
       clearTimeout(debounceTimeout);
       debounceTimeout = setTimeout(() => {
           client.sendQuery(query);
       }, 300);
   }
   ```

### Server-Side Optimizations

1. **Batch Processing**
   - Group multiple document retrievals
   - Use efficient Redis operations
   - Implement connection pooling

2. **Cache Strategies**
   - Implement multi-level caching
   - Use intelligent cache invalidation
   - Pre-compute common queries

3. **Resource Management**
   - Monitor memory usage
   - Implement rate limiting
   - Use connection timeouts

---

## 🔄 **Version History**

### v1.0.0 (Current)
- ✅ WebSocket streaming implementation
- ✅ RAG integration with knowledge base
- ✅ Cache service integration
- ✅ Personalization support
- ✅ Token tracking and cost calculation
- ✅ Source document metadata with page numbers
- ✅ Database tracking
- ✅ Bidirectional communication

### Roadmap
- 🔄 Connection pooling and load balancing
- 🔄 Audio/video streaming support
- 🔄 Multi-language support
- 🔄 Advanced analytics dashboard
- 🔄 Mobile SDK implementation

---

## 📞 **Support & Contact**

For technical support and questions:

- **Documentation:** This file
- **Status Endpoint:** `/ws/status`
- **Logs:** Check server logs with debug mode enabled
- **GitHub:** [Repository Issues](https://github.com/your-repo/issues)

---

**Generated:** December 2024
**Version:** 1.0.0
**Format:** Markdown with Mermaid diagrams
**Target Audience:** Developers integrating WebSocket streaming chat