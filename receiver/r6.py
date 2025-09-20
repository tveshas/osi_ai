import socket
import json
import threading
import time
import random
import math
import hashlib

class AINetworkStack:
    """
    Custom 3-Layer OSI Model with AI
    Layer 3: AI Learning Layer (ML + Federated Learning)
    Layer 2: Smart Transport Layer (Adaptive chunking + reliability)
    Layer 1: Adaptive Network Layer (Discovery + routing optimization)
    """
    
    def __init__(self):
        # Network configuration
        self.port = 8080
        self.discovery_port = 8079
        self.my_ip = self.get_my_ip()
        
        # Layer 1: Adaptive Network Layer
        self.network_layer = AdaptiveNetworkLayer(self.my_ip, self.discovery_port)
        
        # Layer 2: Smart Transport Layer  
        self.transport_layer = SmartTransportLayer()
        
        # Layer 3: AI Learning Layer
        self.ai_layer = AILearningLayer()
        
        # Cross-layer optimization
        self.performance_history = []
        self.adaptive_parameters = {
            'chunk_size': 1000,
            'retry_attempts': 3,
            'learning_rate': 0.01,
            'batch_size': 10
        }
        
        print("🧠 AI-Powered OSI Stack Initialized")
        print(f"📍 Node Address: {self.my_ip}")
        print(f"🔧 Adaptive Parameters: {self.adaptive_parameters}")
    
    def get_my_ip(self):
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))
            ip = temp_sock.getsockname()[0]
            temp_sock.close()
            return ip
        except:
            return "127.0.0.1"

class AdaptiveNetworkLayer:
    """Layer 1: Adaptive Network Layer with AI-driven routing"""
    
    def __init__(self, my_ip, discovery_port):
        self.my_ip = my_ip
        self.discovery_port = discovery_port
        self.peer_nodes = {}  # {ip: {latency, bandwidth, reliability_score}}
        self.routing_table = {}
        self.network_conditions = {'congestion': 0.0, 'packet_loss': 0.0}
        
    def discover_peers(self):
        """AI-enhanced peer discovery with performance profiling"""
        print("🔍 Layer 1: Starting intelligent peer discovery...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.discovery_port))
        sock.settimeout(1.0)
        
        discovered_peers = []
        
        while True:
            try:
                message, sender_address = sock.recvfrom(1024)
                
                if message.decode('utf-8') == "AI_OSI_DISCOVERY":
                    # Measure response time for latency estimation
                    response_start = time.time()
                    
                    # Send enhanced node info
                    node_info = {
                        'type': 'ai_osi_node',
                        'ip': self.my_ip,
                        'port': 8080,
                        'capabilities': ['ml_training', 'adaptive_routing', 'smart_transport'],
                        'performance_score': self.calculate_performance_score(),
                        'timestamp': time.time()
                    }
                    
                    response = json.dumps(node_info)
                    sock.sendto(response.encode('utf-8'), sender_address)
                    
                    response_time = time.time() - response_start
                    
                    # Profile this peer
                    peer_ip = sender_address[0]
                    self.peer_nodes[peer_ip] = {
                        'latency': response_time * 1000,  # ms
                        'bandwidth': 0,  # Will be measured during data transfer
                        'reliability_score': 1.0,  # Will adapt based on success rate
                        'last_seen': time.time()
                    }
                    
                    print(f"📡 Layer 1: Discovered peer {peer_ip} (latency: {response_time*1000:.1f}ms)")
                    discovered_peers.append(peer_ip)
                    
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
        return discovered_peers
    
    def calculate_performance_score(self):
        """Calculate our node's performance score for routing decisions"""
        # Simple scoring based on recent performance
        if not hasattr(self, 'recent_transfers'):
            return 0.8  # Default score
        
        # Score based on successful transfers, speed, etc.
        return min(1.0, 0.5 + len(self.recent_transfers) * 0.1)
    
    def select_best_route(self, target_ip, data_size):
        """AI-driven route selection based on data size and network conditions"""
        if target_ip not in self.peer_nodes:
            return target_ip  # Direct route
        
        peer_info = self.peer_nodes[target_ip]
        
        # Score route based on multiple factors
        latency_score = max(0, 1 - peer_info['latency'] / 1000)  # Lower latency = higher score
        reliability_score = peer_info['reliability_score']
        
        # Adjust for data size (larger data needs more reliable routes)
        if data_size > 10000:
            route_score = 0.3 * latency_score + 0.7 * reliability_score
        else:
            route_score = 0.7 * latency_score + 0.3 * reliability_score
        
        print(f"🧠 Layer 1: Route to {target_ip} score: {route_score:.3f} (latency={peer_info['latency']:.1f}ms, reliability={reliability_score:.3f})")
        
        return target_ip  # For now, always direct route

class SmartTransportLayer:
    """Layer 2: Smart Transport Layer with adaptive chunking and reliability"""
    
    def __init__(self):
        self.active_transfers = {}
        self.chunk_performance = []  # Track chunk size performance
        self.optimal_chunk_size = 1000
        self.congestion_window = 1
        
    def learn_optimal_chunk_size(self, network_conditions):
        """AI learning for optimal chunk size based on network conditions"""
        if len(self.chunk_performance) < 3:
            return self.optimal_chunk_size
        
        # Analyze recent performance
        recent_perf = self.chunk_performance[-10:]
        successful_transfers = [p for p in recent_perf if p['success']]
        
        if not successful_transfers:
            # Network is struggling, reduce chunk size
            self.optimal_chunk_size = max(500, self.optimal_chunk_size * 0.8)
            print(f"🧠 Layer 2: Network issues detected, reducing chunk size to {self.optimal_chunk_size}")
            return self.optimal_chunk_size
        
        # Find best performing chunk size
        best_throughput = 0
        best_size = self.optimal_chunk_size
        
        for perf in successful_transfers:
            throughput = perf['data_size'] / perf['time'] if perf['time'] > 0 else 0
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = perf['chunk_size']
        
        # Gradually adapt towards best size
        adaptation_rate = 0.2
        new_size = int(self.optimal_chunk_size * (1 - adaptation_rate) + best_size * adaptation_rate)
        new_size = max(500, min(5000, new_size))  # Reasonable bounds
        
        if new_size != self.optimal_chunk_size:
            print(f"🧠 Layer 2: Adapting chunk size {self.optimal_chunk_size} → {new_size}")
            self.optimal_chunk_size = new_size
        
        return self.optimal_chunk_size
    
    def chunk_data(self, data, network_conditions):
        """Intelligently chunk data based on learned parameters"""
        chunk_size = self.learn_optimal_chunk_size(network_conditions)
        
        # Adjust for data type
        if isinstance(data, dict):
            if 'gradients' in data or data.get('type') == 'federated_response':
                # ML gradients/responses are small but critical - use smaller, more reliable chunks
                chunk_size = min(chunk_size, 800)
            
            # Convert dict to JSON string for chunking
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        # Create chunks based on character count
        chunk_size_chars = chunk_size * 10  # Convert to approximate character count
        
        if len(data_str) <= chunk_size_chars:
            # Small data, send as single chunk
            chunks = [data_str]
        else:
            # Large data, split into chunks
            chunks = [data_str[i:i+chunk_size_chars] for i in range(0, len(data_str), chunk_size_chars)]
        
        print(f"🧠 Layer 2: Chunked {len(data_str)} chars into {len(chunks)} pieces (chunk_size={chunk_size})")
        return chunks
    
    def send_with_reliability(self, chunks, target_ip, port, sock):
        """Send chunks with adaptive reliability and congestion control"""
        transfer_id = hashlib.md5(f"{time.time()}_{target_ip}".encode()).hexdigest()[:8]
        start_time = time.time()
        successful_chunks = 0
        
        print(f"📦 Layer 2: Starting reliable transfer {transfer_id} to {target_ip}")
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                try:
                    message = {
                        'type': 'smart_chunk',
                        'transfer_id': transfer_id,
                        'chunk_num': i,
                        'total_chunks': len(chunks),
                        'data': chunk,
                        'timestamp': time.time()
                    }
                    
                    sock.sendto(json.dumps(message).encode('utf-8'), (target_ip, port))
                    
                    # Wait for ACK with timeout
                    sock.settimeout(2.0)
                    ack_data = None
                    
                    ack_deadline = time.time() + 2.0
                    while time.time() < ack_deadline:
                        try:
                            response, addr = sock.recvfrom(4096)
                            ack_data = json.loads(response.decode('utf-8'))
                            
                            if (ack_data.get('type') == 'chunk_ack' and 
                                ack_data.get('transfer_id') == transfer_id and
                                ack_data.get('chunk_num') == i):
                                break
                        except socket.timeout:
                            continue
                        except:
                            ack_data = None
                            break
                    
                    if ack_data and ack_data.get('type') == 'chunk_ack':
                        chunk_time = time.time() - chunk_start
                        successful_chunks += 1
                        
                        # Record performance for learning
                        self.chunk_performance.append({
                            'chunk_size': len(str(chunk)),
                            'time': chunk_time,
                            'success': True,
                            'data_size': len(str(chunk)),
                            'attempts': attempts + 1
                        })
                        
                        print(f"✅ Layer 2: Chunk {i+1}/{len(chunks)} sent successfully ({chunk_time:.3f}s, attempt {attempts+1})")
                        break
                    else:
                        attempts += 1
                        print(f"⚠️ Layer 2: Chunk {i+1} failed, retrying... (attempt {attempts+1})")
                        time.sleep(0.1 * attempts)  # Exponential backoff
                
                except Exception as e:
                    attempts += 1
                    print(f"❌ Layer 2: Chunk {i+1} error: {e} (attempt {attempts+1})")
            
            if attempts >= max_attempts:
                print(f"❌ Layer 2: Chunk {i+1} failed after {max_attempts} attempts")
                self.chunk_performance.append({
                    'chunk_size': len(str(chunk)),
                    'time': time.time() - chunk_start,
                    'success': False,
                    'data_size': len(str(chunk)),
                    'attempts': max_attempts
                })
            
            # Adaptive delay between chunks based on network conditions
            time.sleep(0.01)
        
        total_time = time.time() - start_time
        success_rate = successful_chunks / len(chunks)
        
        print(f"📈 Layer 2: Transfer complete - {successful_chunks}/{len(chunks)} chunks ({success_rate:.1%}) in {total_time:.2f}s")
        
        return success_rate > 0.8  # Consider successful if >80% chunks delivered

class AILearningLayer:
    """Layer 3: AI Learning Layer with federated learning and model optimization"""
    
    def __init__(self):
        # Neural network for collaborative learning
        self.weights = [random.uniform(-1, 1) for _ in range(4)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Training data and performance tracking
        self.local_data = self.generate_training_data()
        self.training_history = []
        self.federated_rounds = 0
        
        # AI-driven optimization parameters
        self.adaptive_learning_rate = 0.01
        self.batch_size = 10
        
        print(f"🧠 Layer 3: AI Learning initialized")
        print(f"   Neural network weights: {[round(w, 3) for w in self.weights]}")
        print(f"   Training examples: {len(self.local_data)}")
    
    def generate_training_data(self):
        """Generate training data for our ML model"""
        data = []
        for _ in range(100):
            inputs = [random.uniform(0, 10) for _ in range(4)]
            target = sum(inputs) + random.uniform(-0.5, 0.5)  # Sum with small noise
            data.append({'inputs': inputs, 'target': target})
        return data
    
    def forward_pass(self, inputs):
        """Neural network forward pass"""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 / (1 + math.exp(-max(-500, min(500, weighted_sum)))) * 50
    
    def calculate_gradients(self, batch_data):
        """Calculate gradients for federated learning"""
        total_gradients = {'weights': [0] * 4, 'bias': 0}
        total_loss = 0
        
        for example in batch_data:
            inputs = example['inputs']
            target = example['target']
            
            prediction = self.forward_pass(inputs)
            loss = (prediction - target) ** 2
            total_loss += loss
            
            # Simplified gradient calculation
            error = prediction - target
            for i in range(len(inputs)):
                total_gradients['weights'][i] += error * inputs[i] * 0.01
            total_gradients['bias'] += error * 0.01
        
        # Average gradients
        for i in range(len(total_gradients['weights'])):
            total_gradients['weights'][i] /= len(batch_data)
        total_gradients['bias'] /= len(batch_data)
        
        avg_loss = total_loss / len(batch_data)
        return total_gradients, avg_loss
    
    def federated_learning_step(self, received_gradients, received_loss):
        """Perform one step of federated learning"""
        print(f"🧠 Layer 3: Federated learning step {self.federated_rounds + 1}")
        
        # Calculate our local gradients
        batch_data = random.sample(self.local_data, self.batch_size)
        local_gradients, local_loss = self.calculate_gradients(batch_data)
        
        # Combine gradients (federated averaging)
        combined_gradients = {
            'weights': [
                (local_gradients['weights'][i] + received_gradients['weights'][i]) / 2
                for i in range(len(local_gradients['weights']))
            ],
            'bias': (local_gradients['bias'] + received_gradients['bias']) / 2
        }
        
        # Apply gradients with adaptive learning rate
        for i in range(len(self.weights)):
            self.weights[i] -= self.adaptive_learning_rate * combined_gradients['weights'][i]
        self.bias -= self.adaptive_learning_rate * combined_gradients['bias']
        
        # Adapt learning rate based on loss improvement
        if self.training_history:
            last_loss = self.training_history[-1]['loss']
            if local_loss < last_loss:
                self.adaptive_learning_rate = min(0.05, self.adaptive_learning_rate * 1.01)  # Increase slightly
            else:
                self.adaptive_learning_rate = max(0.001, self.adaptive_learning_rate * 0.99)  # Decrease slightly
        
        # Record training step
        self.training_history.append({
            'round': self.federated_rounds,
            'local_loss': local_loss,
            'remote_loss': received_loss,
            'learning_rate': self.adaptive_learning_rate,
            'weights': self.weights.copy()
        })
        
        self.federated_rounds += 1
        
        print(f"   Local loss: {local_loss:.3f}, Remote loss: {received_loss:.3f}")
        print(f"   Adaptive learning rate: {self.adaptive_learning_rate:.4f}")
        print(f"   Updated weights: {[round(w, 3) for w in self.weights]}")
        
        return local_gradients, local_loss
    
    def evaluate_model(self):
        """Evaluate current model performance"""
        test_data = random.sample(self.local_data, 5)
        total_error = 0
        
        for example in test_data:
            prediction = self.forward_pass(example['inputs'])
            error = abs(prediction - example['target'])
            total_error += error
        
        avg_error = total_error / len(test_data)
        print(f"🎯 Layer 3: Model evaluation - Average error: {avg_error:.3f}")
        return avg_error

class AINetworkReceiver(AINetworkStack):
    """Complete AI-powered OSI receiver combining all three layers"""
    
    def __init__(self):
        super().__init__()
        self.sock = None
        self.discovery_sock = None
        
    def handle_smart_chunk(self, data, sender_address):
        """Layer 2: Handle incoming smart chunks"""
        transfer_id = data['transfer_id']
        chunk_num = data['chunk_num']
        total_chunks = data['total_chunks']
        chunk_data = data['data']
        
        # Initialize transfer tracking
        if transfer_id not in self.transport_layer.active_transfers:
            self.transport_layer.active_transfers[transfer_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'start_time': time.time(),
                'sender': sender_address
            }
            print(f"📦 Layer 2: New transfer {transfer_id} - expecting {total_chunks} chunks")
        
        # Store chunk
        self.transport_layer.active_transfers[transfer_id]['chunks'][chunk_num] = chunk_data
        received_count = len(self.transport_layer.active_transfers[transfer_id]['chunks'])
        
        # Send ACK
        ack = {
            'type': 'chunk_ack',
            'transfer_id': transfer_id,
            'chunk_num': chunk_num,
            'timestamp': time.time()
        }
        self.sock.sendto(json.dumps(ack).encode('utf-8'), sender_address)
        
        print(f"📥 Layer 2: Chunk {chunk_num+1}/{total_chunks} received and ACKed ({received_count}/{total_chunks})")
        
        # Check if transfer complete
        if received_count == total_chunks:
            self.process_complete_transfer(transfer_id)
    
    def process_complete_transfer(self, transfer_id):
        """Process completed data transfer and pass to AI layer"""
        transfer_info = self.transport_layer.active_transfers[transfer_id]
        
        # Reconstruct data
        reconstructed_data = ""
        for i in range(transfer_info['total_chunks']):
            if i in transfer_info['chunks']:
                reconstructed_data += str(transfer_info['chunks'][i])
        
        try:
            # Parse reconstructed data
            data = json.loads(reconstructed_data)
            
            if data.get('type') == 'federated_gradients':
                # Layer 3: Process federated learning data
                received_gradients = data.get('gradients')
                received_loss = data.get('loss')
                
                # Check if data is valid
                if received_gradients is None or received_loss is None:
                    print(f"❌ Layer 3: Invalid federated data - missing gradients or loss")
                    print(f"   Data keys: {list(data.keys())}")
                    return
                
                print(f"🧠 Layer 3: Processing federated learning data")
                print(f"   Received loss: {received_loss:.3f}")
                print(f"   Received gradients: weights={len(received_gradients.get('weights', []))}, bias={'bias' in received_gradients}")
                
                try:
                    local_gradients, local_loss = self.ai_layer.federated_learning_step(received_gradients, received_loss)
                    
                    # Send response back through all layers
                    response_data = {
                        'type': 'federated_response',
                        'gradients': local_gradients,
                        'loss': local_loss,
                        'round': self.ai_layer.federated_rounds,
                        'status': 'success'
                    }
                    
                    print(f"📤 Layer 3: Sending federated response (our loss: {local_loss:.3f})")
                    self.send_intelligent_response(response_data, transfer_info['sender'])
                    
                except Exception as e:
                    print(f"❌ Layer 3: Federated learning failed: {e}")
                    # Send error response
                    error_response = {
                        'type': 'federated_error',
                        'error': str(e),
                        'status': 'failed'
                    }
                    self.send_intelligent_response(error_response, transfer_info['sender'])
                
        except json.JSONDecodeError as e:
            print(f"❌ Layer 2: Could not parse reconstructed data: {e}")
            print(f"   Raw data (first 200 chars): {reconstructed_data[:200]}")
        except Exception as e:
            print(f"❌ Layer 2: Unexpected error processing transfer: {e}")
        
        # Cleanup
        del self.transport_layer.active_transfers[transfer_id]
    
    def send_intelligent_response(self, data, target_address):
        """Send response using all AI-enhanced layers"""
        print(f"📤 Sending intelligent response to {target_address}")
        print(f"   Response type: {data.get('type', 'unknown')}")
        print(f"   Response size: {len(json.dumps(data))} chars")
        
        try:
            # Layer 2: Chunk the response
            chunks = self.transport_layer.chunk_data(data, self.network_layer.network_conditions)
            print(f"📦 Response chunked into {len(chunks)} pieces")
            
            # Layer 2: Send with reliability
            success = self.transport_layer.send_with_reliability(chunks, target_address[0], target_address[1], self.sock)
            
            if success:
                print(f"✅ Intelligent response sent successfully to {target_address}")
            else:
                print(f"❌ Response transmission failed to {target_address}")
                
            return success
            
        except Exception as e:
            print(f"❌ Error sending intelligent response: {e}")
            return False
    
    def start_receiver(self):
        """Main receiver loop processing all layers"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        
        print(f"🚀 AI-Powered OSI Receiver listening on {self.my_ip}:{self.port}")
        print("🧠 All layers active: Network + Transport + AI Learning")
        
        while True:
            try:
                message, sender_address = self.sock.recvfrom(16384)
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    
                    if data.get('type') == 'smart_chunk':
                        self.handle_smart_chunk(data, sender_address)
                    elif data.get('type') == 'layer_test':
                        print(f"🔍 Layer test from {sender_address}")
                        # Simple layer response
                        response = {'type': 'layer_response', 'layers': ['network', 'transport', 'ai']}
                        self.sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                    else:
                        print(f"❓ Unknown message type: {data.get('type')}")
                        
                except json.JSONDecodeError:
                    print(f"📨 Non-JSON message from {sender_address}")
                    
            except KeyboardInterrupt:
                print(f"\n👋 AI OSI Receiver shutting down...")
                print(f"🧠 Final AI state:")
                print(f"   Federated rounds: {self.ai_layer.federated_rounds}")
                print(f"   Final weights: {[round(w, 3) for w in self.ai_layer.weights]}")
                print(f"   Performance history: {len(self.ai_layer.training_history)} records")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if self.sock:
            self.sock.close()
        if self.discovery_sock:
            self.discovery_sock.close()
    
    def run(self):
        """Start the complete AI OSI stack"""
        print(f"\n🎯 Testing initial AI model:")
        self.ai_layer.evaluate_model()
        
        # Start discovery in background
        discovery_thread = threading.Thread(target=self.network_layer.discover_peers)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start main receiver
        self.start_receiver()

if __name__ == "__main__":
    print("🚀 AI-POWERED CUSTOM OSI MODEL - RECEIVER")
    print("="*70)
    print("🏗️  Architecture:")
    print("   Layer 3: AI Learning (Federated ML + Neural Networks)")
    print("   Layer 2: Smart Transport (Adaptive Chunking + Reliability)")  
    print("   Layer 1: Adaptive Network (Intelligent Discovery + Routing)")
    print("="*70)
    
    receiver = AINetworkReceiver()
    receiver.run()