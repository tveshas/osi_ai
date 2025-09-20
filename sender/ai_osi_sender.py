import socket
import json
import time
import random
import math

class AINetworkSender:
    """
    Complete AI-powered OSI sender with custom 3-layer stack
    Demonstrates real distributed ML training across network layers
    """
    
    def __init__(self):
        # Network configuration
        self.discovery_port = 8079
        self.my_ip = self.get_my_ip()
        self.ai_nodes = []
        
        # Initialize our custom OSI stack
        self.network_layer = SenderNetworkLayer(self.discovery_port)
        self.transport_layer = SenderTransportLayer()
        self.ai_layer = SenderAILayer()
        
        # Cross-layer optimization
        self.adaptive_parameters = {
            'chunk_size': 1000,
            'retry_attempts': 3,
            'learning_rate': 0.01,
            'discovery_interval': 30
        }
        
        print("🚀 AI-Powered OSI Sender Initialized")
        print(f"📍 Sender Address: {self.my_ip}")
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

class SenderNetworkLayer:
    """Layer 1: Adaptive Network Layer for sender"""
    
    def __init__(self, discovery_port):
        self.discovery_port = discovery_port
        self.discovered_nodes = {}
        self.routing_performance = {}
        
    def discover_ai_nodes(self):
        """AI-enhanced node discovery with performance profiling"""
        print("🔍 Layer 1: Discovering AI-powered nodes...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(3.0)
        
        # Broadcast discovery message
        discovery_message = "AI_OSI_DISCOVERY"
        networks = ["192.168.1.255", "192.168.0.255", "10.0.0.255", "127.0.0.1"]
        
        for network in networks:
            try:
                discovery_start = time.time()
                sock.sendto(discovery_message.encode('utf-8'), (network, self.discovery_port))
            except:
                continue
        
        # Listen for responses and profile nodes
        discovered_nodes = []
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            try:
                response, address = sock.recvfrom(4096)
                response_time = time.time() - start_time
                
                try:
                    node_info = json.loads(response.decode('utf-8'))
                    
                    if node_info.get('type') == 'ai_osi_node':
                        node_ip = node_info['ip']
                        node_port = node_info['port']
                        capabilities = node_info.get('capabilities', [])
                        performance_score = node_info.get('performance_score', 0.5)
                        
                        # Profile this node
                        self.discovered_nodes[node_ip] = {
                            'ip': node_ip,
                            'port': node_port,
                            'capabilities': capabilities,
                            'performance_score': performance_score,
                            'discovery_latency': response_time * 1000,  # ms
                            'last_seen': time.time(),
                            'reliability_score': 1.0  # Will adapt based on interactions
                        }
                        
                        print(f"✅ Layer 1: Found AI node {node_ip}:{node_port}")
                        print(f"   Capabilities: {', '.join(capabilities)}")
                        print(f"   Performance score: {performance_score:.3f}")
                        print(f"   Discovery latency: {response_time*1000:.1f}ms")
                        
                        discovered_nodes.append(node_info)
                
                except json.JSONDecodeError:
                    continue
                        
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
        
        print(f"🌐 Layer 1: Discovery complete - found {len(discovered_nodes)} AI nodes")
        return discovered_nodes
    
    def select_optimal_node(self, data_type, data_size):
        """AI-driven node selection based on data characteristics"""
        if not self.discovered_nodes:
            return None
        
        best_node = None
        best_score = 0
        
        for node_ip, node_info in self.discovered_nodes.items():
            # Calculate suitability score
            score = 0
            
            # Performance-based scoring
            score += node_info['performance_score'] * 0.4
            score += node_info['reliability_score'] * 0.3
            
            # Latency-based scoring (lower latency = higher score)
            latency_score = max(0, 1 - node_info['discovery_latency'] / 1000)
            score += latency_score * 0.2
            
            # Capability matching
            if data_type == 'ml_training' and 'ml_training' in node_info['capabilities']:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_node = node_info
        
        if best_node:
            print(f"🧠 Layer 1: Selected optimal node {best_node['ip']} (score: {best_score:.3f})")
        
        return best_node

class SenderTransportLayer:
    """Layer 2: Smart Transport Layer for sender"""
    
    def __init__(self):
        self.chunk_performance = []
        self.optimal_chunk_size = 1000
        self.congestion_control = {'window_size': 1, 'threshold': 4}
        
    def adaptive_chunk_sizing(self, data, target_node):
        """AI-driven adaptive chunk sizing based on target node and data type"""
        base_size = self.optimal_chunk_size
        
        # Adjust based on target node performance
        if target_node and 'performance_score' in target_node:
            performance_factor = target_node['performance_score']
            base_size = int(base_size * (0.5 + performance_factor))
        
        # Adjust based on data type
        if isinstance(data, dict) and 'gradients' in data:
            # ML gradients are critical - use smaller, more reliable chunks
            base_size = min(base_size, 800)
        
        # Learn from historical performance
        if len(self.chunk_performance) >= 5:
            recent_perf = self.chunk_performance[-5:]
            avg_success_rate = sum(1 for p in recent_perf if p['success']) / len(recent_perf)
            
            if avg_success_rate < 0.8:
                # Network struggling, reduce chunk size
                base_size = int(base_size * 0.8)
                print(f"🧠 Layer 2: Reducing chunk size due to low success rate ({avg_success_rate:.1%})")
            elif avg_success_rate > 0.95:
                # Network performing well, can increase chunk size
                base_size = int(base_size * 1.1)
                print(f"🧠 Layer 2: Increasing chunk size due to high success rate ({avg_success_rate:.1%})")
        
        # Keep within reasonable bounds
        self.optimal_chunk_size = max(500, min(3000, base_size))
        
        print(f"🧠 Layer 2: Adaptive chunk size: {self.optimal_chunk_size}")
        return self.optimal_chunk_size
    
    def intelligent_send(self, data, target_node, sock):
        """Send data using intelligent transport protocols"""
        chunk_size = self.adaptive_chunk_sizing(data, target_node)
        
        # Serialize data
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        # Create chunks
        chunks = [data_str[i:i+chunk_size*10] for i in range(0, len(data_str), chunk_size*10)]
        
        print(f"📦 Layer 2: Sending {len(data_str)} chars in {len(chunks)} intelligent chunks")
        
        # Send with adaptive reliability
        return self.send_chunks_with_intelligence(chunks, target_node, sock)
    
    def send_chunks_with_intelligence(self, chunks, target_node, sock):
        """Send chunks with AI-enhanced reliability and congestion control"""
        transfer_id = f"ai_transfer_{int(time.time()*1000)}"
        start_time = time.time()
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            success = False
            
            # Adaptive retry logic
            max_attempts = 3 if target_node.get('reliability_score', 1.0) > 0.8 else 5
            
            for attempt in range(max_attempts):
                try:
                    message = {
                        'type': 'smart_chunk',
                        'transfer_id': transfer_id,
                        'chunk_num': i,
                        'total_chunks': len(chunks),
                        'data': chunk,
                        'timestamp': time.time(),
                        'sender_performance': self.get_sender_metrics()
                    }
                    
                    # Send chunk
                    sock.sendto(json.dumps(message).encode('utf-8'), 
                              (target_node['ip'], target_node['port']))
                    
                    # Wait for ACK with adaptive timeout
                    ack_timeout = 1.0 + (target_node.get('discovery_latency', 100) / 1000)
                    sock.settimeout(ack_timeout)
                    
                    # Listen for ACK
                    ack_received = False
                    ack_deadline = time.time() + ack_timeout
                    
                    while time.time() < ack_deadline:
                        try:
                            response, addr = sock.recvfrom(4096)
                            ack_data = json.loads(response.decode('utf-8'))
                            
                            if (ack_data.get('type') == 'chunk_ack' and 
                                ack_data.get('transfer_id') == transfer_id and
                                ack_data.get('chunk_num') == i):
                                ack_received = True
                                break
                        except socket.timeout:
                            break
                        except:
                            continue
                    
                    if ack_received:
                        chunk_time = time.time() - chunk_start
                        successful_chunks += 1
                        success = True
                        
                        # Record successful performance
                        self.chunk_performance.append({
                            'chunk_size': len(chunk),
                            'time': chunk_time,
                            'success': True,
                            'attempts': attempt + 1,
                            'target_node': target_node['ip']
                        })
                        
                        print(f"✅ Layer 2: Chunk {i+1}/{len(chunks)} delivered "
                              f"(attempt {attempt+1}, {chunk_time:.3f}s)")
                        break
                    else:
                        print(f"⚠️ Layer 2: Chunk {i+1} ACK timeout, retrying...")
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
                except Exception as e:
                    print(f"❌ Layer 2: Chunk {i+1} error: {e}")
                    time.sleep(0.1 * (attempt + 1))
            
            if not success:
                # Record failed performance
                self.chunk_performance.append({
                    'chunk_size': len(chunk),
                    'time': time.time() - chunk_start,
                    'success': False,
                    'attempts': max_attempts,
                    'target_node': target_node['ip']
                })
                print(f"❌ Layer 2: Chunk {i+1} failed after {max_attempts} attempts")
        
        total_time = time.time() - start_time
        success_rate = successful_chunks / len(chunks)
        
        # Update target node reliability score
        if target_node['ip'] in [p['target_node'] for p in self.chunk_performance[-10:]]:
            recent_successes = [p for p in self.chunk_performance[-10:] 
                              if p['target_node'] == target_node['ip'] and p['success']]
            recent_attempts = [p for p in self.chunk_performance[-10:] 
                             if p['target_node'] == target_node['ip']]
            
            if recent_attempts:
                new_reliability = len(recent_successes) / len(recent_attempts)
                target_node['reliability_score'] = (target_node['reliability_score'] * 0.7 + 
                                                   new_reliability * 0.3)
        
        print(f"📈 Layer 2: Transfer summary - {successful_chunks}/{len(chunks)} chunks "
              f"({success_rate:.1%}) in {total_time:.2f}s")
        
        return success_rate > 0.8
    
    def get_sender_metrics(self):
        """Get current sender performance metrics"""
        if not self.chunk_performance:
            return {'avg_time': 0, 'success_rate': 1.0}
        
        recent_perf = self.chunk_performance[-10:]
        avg_time = sum(p['time'] for p in recent_perf) / len(recent_perf)
        success_rate = sum(1 for p in recent_perf if p['success']) / len(recent_perf)
        
        return {'avg_time': avg_time, 'success_rate': success_rate}

class SenderAILayer:
    """Layer 3: AI Learning Layer for sender"""
    
    def __init__(self):
        # Neural network (same architecture as receiver for compatibility)
        self.weights = [random.uniform(-1, 1) for _ in range(4)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Training data (different from receiver for diversity)
        self.local_data = self.generate_diverse_training_data()
        self.federated_history = []
        self.training_rounds = 0
        
        print(f"🧠 Layer 3: AI Learning initialized")
        print(f"   Weights: {[round(w, 3) for w in self.weights]}")
        print(f"   Training data: {len(self.local_data)} examples")
    
    def generate_diverse_training_data(self):
        """Generate training data with different characteristics than receiver"""
        data = []
        for _ in range(100):
            # Different range and pattern for diversity
            inputs = [random.uniform(1, 8) for _ in range(4)]
            # Slightly different target function for generalization
            target = sum(inputs) * 1.1 + random.uniform(-1, 1)
            data.append({'inputs': inputs, 'target': target})
        
        return data
    
    def forward_pass(self, inputs):
        """Neural network forward pass"""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 / (1 + math.exp(-max(-500, min(500, weighted_sum)))) * 50
    
    def calculate_local_gradients(self, batch_size=10):
        """Calculate gradients from local training data"""
        batch = random.sample(self.local_data, min(batch_size, len(self.local_data)))
        
        total_gradients = {'weights': [0] * 4, 'bias': 0}
        total_loss = 0
        
        for example in batch:
            inputs = example['inputs']
            target = example['target']
            
            prediction = self.forward_pass(inputs)
            loss = (prediction - target) ** 2
            total_loss += loss
            
            # Calculate gradients
            error = prediction - target
            for i in range(len(inputs)):
                total_gradients['weights'][i] += error * inputs[i] * 0.01
            total_gradients['bias'] += error * 0.01
        
        # Average gradients
        for i in range(len(total_gradients['weights'])):
            total_gradients['weights'][i] /= len(batch)
        total_gradients['bias'] /= len(batch)
        
        avg_loss = total_loss / len(batch)
        return total_gradients, avg_loss
    
    def prepare_federated_data(self):
        """Prepare data for federated learning transmission"""
        gradients, loss = self.calculate_local_gradients()
        
        federated_data = {
            'type': 'federated_gradients',
            'gradients': gradients,
            'loss': loss,
            'round': self.training_rounds,
            'sender_id': 'ai_sender',
            'timestamp': time.time()
        }
        
        print(f"🧠 Layer 3: Prepared federated data (loss: {loss:.3f})")
        return federated_data
    
    def process_federated_response(self, response_data):
        """Process federated learning response and update model"""
        if not response_data or response_data.get('status') == 'failed':
            print(f"❌ Layer 3: Federated learning failed on remote side")
            return False
            
        received_gradients = response_data.get('gradients')
        received_loss = response_data.get('loss')
        
        if not received_gradients or received_loss is None:
            print(f"❌ Layer 3: Invalid response data - missing gradients or loss")
            print(f"   Response keys: {list(response_data.keys()) if response_data else 'None'}")
            return False
        
        # Calculate our local gradients
        local_gradients, local_loss = self.calculate_local_gradients()
        
        # Combine gradients (federated averaging)
        try:
            combined_gradients = {
                'weights': [
                    (local_gradients['weights'][i] + received_gradients['weights'][i]) / 2
                    for i in range(len(local_gradients['weights']))
                ],
                'bias': (local_gradients['bias'] + received_gradients['bias']) / 2
            }
            
            # Apply combined gradients
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * combined_gradients['weights'][i]
            self.bias -= self.learning_rate * combined_gradients['bias']
            
            # Record federated learning step
            self.federated_history.append({
                'round': self.training_rounds,
                'local_loss': local_loss,
                'remote_loss': received_loss,
                'combined_gradients': combined_gradients,
                'weights_after': self.weights.copy()
            })
            
            self.training_rounds += 1
            
            print(f"🧠 Layer 3: Federated learning step completed")
            print(f"   Round: {self.training_rounds}")
            print(f"   Local loss: {local_loss:.3f}, Remote loss: {received_loss:.3f}")
            print(f"   Updated weights: {[round(w, 3) for w in self.weights]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Layer 3: Error processing gradients: {e}")
            return False
    
    def evaluate_model_performance(self):
        """Evaluate current model performance"""
        test_batch = random.sample(self.local_data, 10)
        total_error = 0
        
        for example in test_batch:
            prediction = self.forward_pass(example['inputs'])
            error = abs(prediction - example['target'])
            total_error += error
        
        avg_error = total_error / len(test_batch)
        print(f"🎯 Layer 3: Model performance - Average error: {avg_error:.3f}")
        return avg_error

class AINetworkSenderComplete(AINetworkSender):
    """Complete AI OSI Sender with all three layers integrated"""
    
    def __init__(self):
        super().__init__()
    
    def send_federated_learning_data(self, target_node):
        """Send federated learning data through all AI layers"""
        print(f"\n🚀 Initiating federated learning with {target_node['ip']}")
        print("="*60)
        
        # Layer 3: Prepare AI data
        federated_data = self.ai_layer.prepare_federated_data()
        
        # Layer 1: Select optimal route (already done)
        print(f"🧠 Layer 1: Route selected - direct to {target_node['ip']}")
        
        # Layer 2: Send through smart transport
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10.0)
        
        try:
            success = self.transport_layer.intelligent_send(federated_data, target_node, sock)
            
            if success:
                print(f"✅ Federated data sent successfully")
                
                # Wait for federated response with better handling
                print(f"⏳ Waiting for federated learning response...")
                
                # Give more time for response and better error handling
                response_data = self.wait_for_federated_response_improved(sock, target_node)
                
                if response_data:
                    # Layer 3: Process response
                    success = self.ai_layer.process_federated_response(response_data)
                    if success:
                        print(f"✅ Federated learning round completed successfully!")
                        return True
                    else:
                        print(f"❌ Failed to process federated response")
                        return False
                else:
                    print(f"❌ No valid response received from {target_node['ip']}")
                    return False
            else:
                print(f"❌ Failed to send federated data to {target_node['ip']}")
                return False
                
        except Exception as e:
            print(f"❌ Federated learning error: {e}")
            return False
        finally:
            sock.close()
    
    def wait_for_federated_response_improved(self, sock, target_node):
        """Improved response waiting with better timeout handling"""
        response_chunks = {}
        total_chunks = None
        
        # Longer timeout for response
        timeout = time.time() + 20.0  # 20 second timeout
        
        print(f"🔍 Listening for response from {target_node['ip']}...")
        
        while time.time() < timeout:
            try:
                # Shorter socket timeout but longer overall timeout
                sock.settimeout(1.0)
                message, addr = sock.recvfrom(16384)
                
                print(f"📨 Received message from {addr[0]} (expecting from {target_node['ip']})")
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    msg_type = data.get('type')
                    
                    print(f"📨 Message type: {msg_type}")
                    
                    if msg_type == 'smart_chunk':
                        chunk_num = data['chunk_num']
                        total_chunks = data['total_chunks']
                        transfer_id = data.get('transfer_id', 'unknown')
                        response_chunks[chunk_num] = data['data']
                        
                        # Send ACK
                        ack = {
                            'type': 'chunk_ack',
                            'transfer_id': transfer_id,
                            'chunk_num': chunk_num,
                            'timestamp': time.time()
                        }
                        sock.sendto(json.dumps(ack).encode('utf-8'), addr)
                        
                        print(f"📥 Response chunk {chunk_num+1}/{total_chunks} received and ACKed")
                        
                        # Check if complete
                        if len(response_chunks) == total_chunks:
                            print(f"✅ All {total_chunks} response chunks received, reconstructing...")
                            
                            # Reconstruct response
                            full_response = ""
                            for i in range(total_chunks):
                                if i in response_chunks:
                                    full_response += str(response_chunks[i])
                            
                            print(f"📝 Reconstructed response length: {len(full_response)} chars")
                            print(f"📝 Response preview: {full_response[:100]}...")
                            
                            try:
                                parsed_response = json.loads(full_response)
                                response_type = parsed_response.get('type', 'unknown')
                                
                                print(f"✅ Successfully parsed response type: {response_type}")
                                
                                # Check response type
                                if response_type == 'federated_error':
                                    print(f"❌ Received error response: {parsed_response.get('error')}")
                                    return None
                                elif response_type == 'federated_response':
                                    print(f"✅ Received valid federated response!")
                                    print(f"   Response loss: {parsed_response.get('loss', 'N/A')}")
                                    print(f"   Response round: {parsed_response.get('round', 'N/A')}")
                                    return parsed_response
                                else:
                                    print(f"❓ Unknown response type: {response_type}")
                                    print(f"   Available keys: {list(parsed_response.keys())}")
                                    return parsed_response
                                    
                            except json.JSONDecodeError as e:
                                print(f"❌ Could not parse reconstructed response: {e}")
                                print(f"   Raw response (first 300 chars): {full_response[:300]}")
                                return None
                    else:
                        print(f"📨 Ignoring message type: {msg_type}")
                        continue
                
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"   Raw message: {message.decode('utf-8', errors='replace')[:100]}")
                    continue
                
            except socket.timeout:
                # This is normal, just continue listening
                remaining_time = timeout - time.time()
                if remaining_time > 0:
                    print(f"⏳ Still waiting... {remaining_time:.1f}s remaining")
                continue
            except Exception as e:
                print(f"❌ Unexpected error while waiting for response: {e}")
                continue
        
        print(f"⏰ Response timeout after 20 seconds")
        print(f"   Received {len(response_chunks)} chunks out of {total_chunks or 'unknown'}")
        return None
    
    def wait_for_federated_response(self, sock, target_node):
        """Wait for and reconstruct federated learning response"""
        response_chunks = {}
        total_chunks = None
        
        timeout = time.time() + 15.0  # 15 second timeout
        
        while time.time() < timeout:
            try:
                sock.settimeout(2.0)
                message, addr = sock.recvfrom(16384)
                data = json.loads(message.decode('utf-8'))
                
                if data.get('type') == 'smart_chunk':
                    chunk_num = data['chunk_num']
                    total_chunks = data['total_chunks']
                    response_chunks[chunk_num] = data['data']
                    
                    # Send ACK
                    ack = {
                        'type': 'chunk_ack',
                        'transfer_id': data['transfer_id'],
                        'chunk_num': chunk_num
                    }
                    sock.sendto(json.dumps(ack).encode('utf-8'), addr)
                    
                    print(f"📨 Response chunk {chunk_num+1}/{total_chunks} received")
                    
                    # Check if complete
                    if len(response_chunks) == total_chunks:
                        # Reconstruct response
                        full_response = ""
                        for i in range(total_chunks):
                            if i in response_chunks:
                                full_response += response_chunks[i]
                        
                        try:
                            parsed_response = json.loads(full_response)
                            
                            # Check if it's an error response
                            if parsed_response.get('type') == 'federated_error':
                                print(f"❌ Received error response: {parsed_response.get('error')}")
                                return None
                            elif parsed_response.get('type') == 'federated_response':
                                print(f"✅ Received valid federated response")
                                return parsed_response
                            else:
                                print(f"❓ Unknown response type: {parsed_response.get('type')}")
                                return parsed_response
                                
                        except json.JSONDecodeError as e:
                            print(f"❌ Could not parse federated response: {e}")
                            print(f"   Raw response (first 200 chars): {full_response[:200]}")
                            return None
                
            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in response: {e}")
                continue
            except Exception as e:
                print(f"❌ Response error: {e}")
                break
        
        print("⏰ Federated response timeout")
        return None
    
    def run_multiple_federated_rounds(self, rounds=3):
        """Run multiple rounds of federated learning"""
        if not self.ai_nodes:
            print("❌ No AI nodes available for federated learning")
            return
        
        print(f"\n🚀 Starting {rounds} rounds of federated learning")
        print("="*60)
        
        initial_performance = self.ai_layer.evaluate_model_performance()
        
        successful_rounds = 0
        
        for round_num in range(rounds):
            print(f"\n📍 FEDERATED ROUND {round_num + 1}/{rounds}")
            print("-" * 40)
            
            # Select best node for this round
            best_node = self.network_layer.select_optimal_node('ml_training', 1000)
            
            if best_node:
                success = self.send_federated_learning_data(best_node)
                if success:
                    successful_rounds += 1
                    print(f"✅ Round {round_num + 1} completed successfully")
                else:
                    print(f"❌ Round {round_num + 1} failed")
            else:
                print(f"❌ No suitable node found for round {round_num + 1}")
            
            time.sleep(2)  # Delay between rounds
        
        final_performance = self.ai_layer.evaluate_model_performance()
        
        print(f"\n🎯 FEDERATED LEARNING SUMMARY")
        print("="*50)
        print(f"   Rounds attempted: {rounds}")
        print(f"   Rounds successful: {successful_rounds}")
        print(f"   Success rate: {successful_rounds/rounds:.1%}")
        print(f"   Initial model error: {initial_performance:.3f}")
        print(f"   Final model error: {final_performance:.3f}")
        print(f"   Improvement: {initial_performance - final_performance:.3f}")
        print(f"   Total federated rounds: {self.ai_layer.training_rounds}")
    
    def interactive_mode(self):
        """Interactive menu for AI OSI operations"""
        while True:
            print("\n" + "="*70)
            print("🧠 AI-POWERED CUSTOM OSI MODEL - SENDER")
            print(f"🎯 Federated rounds completed: {self.ai_layer.training_rounds}")
            print(f"🌐 Connected AI nodes: {len(self.ai_nodes)}")
            print(f"📊 Transport performance: {len(self.transport_layer.chunk_performance)} transfers")
            print("="*70)
            print("1️⃣  Test current AI model")
            print("2️⃣  Single federated learning round")
            print("3️⃣  Multiple federated rounds (3 rounds)")
            print("4️⃣  Stress test (10 rounds)")
            print("5️⃣  Show network layer status")
            print("6️⃣  Show transport layer stats")
            print("7️⃣  Show AI layer progress")
            print("8️⃣  Discover AI nodes again")
            print("9️⃣  Quit")
            
            choice = input("Choose (1-9): ")
            
            if choice == "1":
                self.ai_layer.evaluate_model_performance()
            elif choice == "2":
                self.run_multiple_federated_rounds(1)
            elif choice == "3":
                self.run_multiple_federated_rounds(3)
            elif choice == "4":
                self.run_multiple_federated_rounds(10)
            elif choice == "5":
                self.show_network_status()
            elif choice == "6":
                self.show_transport_stats()
            elif choice == "7":
                self.show_ai_progress()
            elif choice == "8":
                self.discover_and_connect()
            elif choice == "9":
                print("👋 AI OSI Sender shutting down!")
                break
            else:
                print("❌ Invalid choice")
    
    def show_network_status(self):
        """Show Layer 1 network status"""
        print(f"\n🌐 LAYER 1: NETWORK STATUS")
        print("="*40)
        print(f"Discovered nodes: {len(self.network_layer.discovered_nodes)}")
        
        for ip, node in self.network_layer.discovered_nodes.items():
            print(f"  📍 {ip}:{node['port']}")
            print(f"     Performance: {node['performance_score']:.3f}")
            print(f"     Reliability: {node['reliability_score']:.3f}")
            print(f"     Latency: {node['discovery_latency']:.1f}ms")
    
    def show_transport_stats(self):
        """Show Layer 2 transport statistics"""
        print(f"\n📦 LAYER 2: TRANSPORT STATISTICS")
        print("="*40)
        
        if not self.transport_layer.chunk_performance:
            print("No transport data yet")
            return
        
        recent_perf = self.transport_layer.chunk_performance[-20:]
        success_rate = sum(1 for p in recent_perf if p['success']) / len(recent_perf)
        avg_time = sum(p['time'] for p in recent_perf) / len(recent_perf)
        
        print(f"Recent performance (last 20 transfers):")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average time per chunk: {avg_time:.3f}s")
        print(f"  Current optimal chunk size: {self.transport_layer.optimal_chunk_size}")
        print(f"  Total transfers: {len(self.transport_layer.chunk_performance)}")
    
    def show_ai_progress(self):
        """Show Layer 3 AI learning progress"""
        print(f"\n🧠 LAYER 3: AI LEARNING PROGRESS")
        print("="*40)
        print(f"Current weights: {[round(w, 3) for w in self.ai_layer.weights]}")
        print(f"Current bias: {round(self.ai_layer.bias, 3)}")
        print(f"Learning rate: {self.ai_layer.learning_rate}")
        print(f"Federated rounds: {self.ai_layer.training_rounds}")
        print(f"Training history: {len(self.ai_layer.federated_history)} records")
        
        if self.ai_layer.federated_history:
            print(f"\nRecent federated learning:")
            for record in self.ai_layer.federated_history[-3:]:
                print(f"  Round {record['round']}: local_loss={record['local_loss']:.3f}, "
                      f"remote_loss={record['remote_loss']:.3f}")
    
    def discover_and_connect(self):
        """Discover and connect to AI nodes"""
        discovered_nodes = self.network_layer.discover_ai_nodes()
        self.ai_nodes = [node for node in discovered_nodes 
                        if 'ml_training' in node.get('capabilities', [])]
        
        print(f"✅ Connected to {len(self.ai_nodes)} AI-capable nodes")

if __name__ == "__main__":
    print("🚀 AI-POWERED CUSTOM OSI MODEL - SENDER")
    print("="*70)
    print("🏗️  Architecture:")
    print("   Layer 3: AI Learning (Federated ML + Neural Networks)")
    print("   Layer 2: Smart Transport (Adaptive Chunking + Reliability)")
    print("   Layer 1: Adaptive Network (Intelligent Discovery + Routing)")
    print("="*70)
    
    sender = AINetworkSenderComplete()
    
    # Initial setup
    print("\n🔍 Testing initial AI model:")
    sender.ai_layer.evaluate_model_performance()
    
    # Discover nodes
    sender.discover_and_connect()
    
    if sender.ai_nodes:
        sender.interactive_mode()
    else:
        print("😢 No AI nodes found for federated learning")
        print("💡 Make sure ai_osi_receiver.py is running on another laptop")