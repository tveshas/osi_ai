import socket
import json
import random
import math
import time

class SimpleFederatedSender:
    def __init__(self):
        self.discovery_port = 8079
        self.my_ip = self.get_my_ip()
        self.federated_nodes = []
        
        # Simple neural network - same structure as receiver
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Our training data (different from receiver)
        self.training_data = self.generate_training_data()
        self.training_rounds = 0
        
        print("🚀 Simple Federated Sender")
        print(f"📍 IP: {self.my_ip}")
        print(f"🎯 Initial weights: {[round(w, 3) for w in self.weights]}")
        print(f"🎯 Initial bias: {round(self.bias, 3)}")
        print(f"📊 Training data: {len(self.training_data)} examples")
    
    def get_my_ip(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def generate_training_data(self):
        """Generate different training data for diversity"""
        data = []
        for _ in range(50):  # 50 examples
            # Slightly different range for diversity
            inputs = [random.uniform(1, 6) for _ in range(3)]
            target = sum(inputs) + random.uniform(-0.5, 0.5)  # Sum with noise
            data.append({"inputs": inputs, "target": target})
        
        print(f"📚 Generated training data example:")
        print(f"   Input: {[round(x, 2) for x in data[0]['inputs']]} → Target: {round(data[0]['target'], 2)}")
        return data
    
    def forward_pass(self, inputs):
        """Simple neural network: output = sum(weights * inputs) + bias"""
        result = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return max(0, result)  # Simple ReLU activation
    
    def calculate_gradients(self):
        """Calculate gradients from our local data"""
        total_gradients = {"weights": [0, 0, 0], "bias": 0}
        total_loss = 0
        batch_size = 10  # Use 10 examples
        
        batch = random.sample(self.training_data, min(batch_size, len(self.training_data)))
        
        for example in batch:
            inputs = example["inputs"]
            target = example["target"]
            
            # Forward pass
            prediction = self.forward_pass(inputs)
            
            # Calculate loss
            loss = (prediction - target) ** 2
            total_loss += loss
            
            # Simple gradient calculation
            error = prediction - target
            for i in range(3):
                total_gradients["weights"][i] += error * inputs[i] / batch_size
            total_gradients["bias"] += error / batch_size
        
        avg_loss = total_loss / batch_size
        return total_gradients, avg_loss
    
    def apply_gradients(self, gradients):
        """Apply gradients to update our model"""
        for i in range(3):
            self.weights[i] -= self.learning_rate * gradients["weights"][i]
        self.bias -= self.learning_rate * gradients["bias"]
    
    def test_model(self):
        """Test our current model"""
        test_batch = random.sample(self.training_data, 5)
        total_error = 0
        
        print(f"\n🧪 Testing our model:")
        for i, example in enumerate(test_batch):
            prediction = self.forward_pass(example["inputs"])
            actual = example["target"]
            error = abs(prediction - actual)
            total_error += error
            
            print(f"   Test {i+1}: {[round(x, 1) for x in example['inputs']]} → "
                  f"predicted {prediction:.2f}, actual {actual:.2f}, error {error:.2f}")
        
        avg_error = total_error / len(test_batch)
        print(f"📊 Our average error: {avg_error:.3f}")
        return avg_error
    
    def discover_federated_nodes(self):
        """Find federated learning nodes"""
        print("🔍 Looking for federated learning nodes...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(3.0)
        
        # Broadcast discovery
        message = "FIND_FEDERATED_NODES"
        networks = ["192.168.1.255", "192.168.0.255", "127.0.0.1"]
        
        for network in networks:
            try:
                sock.sendto(message.encode('utf-8'), (network, self.discovery_port))
            except:
                pass
        
        # Listen for responses
        found_nodes = []
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            try:
                response, address = sock.recvfrom(1024)
                node_info = json.loads(response.decode('utf-8'))
                
                if node_info.get("type") == "federated_node":
                    node_ip = node_info["ip"]
                    node_port = node_info["port"]
                    training_rounds = node_info.get("training_rounds", 0)
                    
                    if node_ip != self.my_ip:  # Don't connect to ourselves
                        found_nodes.append({
                            "ip": node_ip,
                            "port": node_port,
                            "training_rounds": training_rounds
                        })
                        print(f"✅ Found federated node: {node_ip}:{node_port} (rounds: {training_rounds})")
                
            except socket.timeout:
                continue
            except:
                continue
        
        sock.close()
        self.federated_nodes = found_nodes
        
        if found_nodes:
            print(f"🌐 Connected to {len(found_nodes)} federated nodes")
        else:
            print("❌ No federated nodes found")
            manual_ip = input("💻 Enter receiver IP manually (or press Enter): ")
            if manual_ip.strip():
                self.federated_nodes = [{"ip": manual_ip.strip(), "port": 8080, "training_rounds": 0}]
        
        return len(self.federated_nodes) > 0
    
    def do_federated_learning(self, target_node):
        """Do one round of federated learning with a node"""
        print(f"\n🤝 Starting federated learning with {target_node['ip']}")
        print("="*50)
        
        # Calculate our gradients
        our_gradients, our_loss = self.calculate_gradients()
        print(f"💻 Our gradients: {[round(g, 4) for g in our_gradients['weights']]}")
        print(f"💻 Our loss: {our_loss:.3f}")
        
        # Prepare request
        request = {
            "type": "federated_request",
            "gradients": our_gradients,
            "loss": our_loss,
            "round": self.training_rounds
        }
        
        # Send to target node
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10.0)
        
        try:
            print(f"📤 Sending gradients to {target_node['ip']}...")
            sock.sendto(json.dumps(request).encode('utf-8'), (target_node['ip'], target_node['port']))
            
            # Wait for response
            print(f"⏳ Waiting for response...")
            response, addr = sock.recvfrom(4096)
            response_data = json.loads(response.decode('utf-8'))
            
            if response_data.get("type") == "federated_response":
                received_gradients = response_data["gradients"]
                received_loss = response_data["loss"]
                
                print(f"📨 Received gradients: {[round(g, 4) for g in received_gradients['weights']]}")
                print(f"📨 Received loss: {received_loss:.3f}")
                
                # Average gradients (federated learning!)
                combined_gradients = {
                    "weights": [
                        (our_gradients["weights"][i] + received_gradients["weights"][i]) / 2
                        for i in range(3)
                    ],
                    "bias": (our_gradients["bias"] + received_gradients["bias"]) / 2
                }
                
                print(f"🔄 Combined gradients: {[round(g, 4) for g in combined_gradients['weights']]}")
                
                # Update our model
                old_weights = self.weights.copy()
                self.apply_gradients(combined_gradients)
                
                print(f"⚖️ Our weights: {[round(w, 3) for w in old_weights]} → {[round(w, 3) for w in self.weights]}")
                
                self.training_rounds += 1
                print(f"✅ Federated learning round {self.training_rounds} completed!")
                
                return True
            else:
                print(f"❌ Unexpected response: {response_data.get('type')}")
                return False
                
        except socket.timeout:
            print(f"⏰ Timeout waiting for response from {target_node['ip']}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            sock.close()
    
    def compare_models(self, target_node):
        """Compare our model with the target node's model"""
        print(f"\n📊 Comparing models with {target_node['ip']}")
        
        # Test our model
        our_error = self.test_model()
        
        # Get their model stats
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        try:
            request = {"type": "model_test"}
            sock.sendto(json.dumps(request).encode('utf-8'), (target_node['ip'], target_node['port']))
            
            response, addr = sock.recvfrom(4096)
            their_stats = json.loads(response.decode('utf-8'))
            
            if their_stats.get("type") == "model_stats":
                their_error = their_stats["error"]
                their_weights = their_stats["weights"]
                their_rounds = their_stats["rounds"]
                
                print(f"\n📈 MODEL COMPARISON:")
                print(f"   Our error: {our_error:.3f}")
                print(f"   Their error: {their_error:.3f}")
                print(f"   Difference: {abs(our_error - their_error):.3f}")
                print(f"\n🧠 WEIGHT COMPARISON:")
                print(f"   Our weights: {[round(w, 3) for w in self.weights]}")
                print(f"   Their weights: {[round(w, 3) for w in their_weights]}")
                print(f"\n🏁 TRAINING PROGRESS:")
                print(f"   Our rounds: {self.training_rounds}")
                print(f"   Their rounds: {their_rounds}")
                
                return their_stats
            
        except Exception as e:
            print(f"❌ Could not get their model stats: {e}")
        finally:
            sock.close()
        
        return None
    
    def train_multiple_rounds(self, rounds=5):
        """Train for multiple rounds"""
        if not self.federated_nodes:
            print("❌ No federated nodes available")
            return
        
        print(f"\n🚀 Starting {rounds} rounds of federated learning")
        print("="*60)
        
        initial_error = self.test_model()
        successful_rounds = 0
        
        for round_num in range(rounds):
            print(f"\n📍 ROUND {round_num + 1}/{rounds}")
            
            # Use first available node
            target_node = self.federated_nodes[0]
            success = self.do_federated_learning(target_node)
            
            if success:
                successful_rounds += 1
                # Test model after each round
                self.test_model()
            
            time.sleep(1)  # Small delay between rounds
        
        final_error = self.test_model()
        
        print(f"\n🏆 TRAINING SUMMARY")
        print("="*40)
        print(f"   Rounds attempted: {rounds}")
        print(f"   Rounds successful: {successful_rounds}")
        print(f"   Success rate: {successful_rounds/rounds:.1%}")
        print(f"   Initial error: {initial_error:.3f}")
        print(f"   Final error: {final_error:.3f}")
        print(f"   Improvement: {initial_error - final_error:.3f}")
    
    def interactive_mode(self):
        """Simple interactive menu"""
        while True:
            print("\n" + "="*50)
            print("🤝 SIMPLE FEDERATED LEARNING")
            print(f"🎯 Training rounds: {self.training_rounds}")
            print(f"🌐 Connected nodes: {len(self.federated_nodes)}")
            print("="*50)
            print("1️⃣  Test our current model")
            print("2️⃣  Single federated learning round")
            print("3️⃣  Multiple rounds (5 rounds)")
            print("4️⃣  Compare models")
            print("5️⃣  Find nodes again")
            print("6️⃣  Quit")
            
            choice = input("Choose (1-6): ")
            
            if choice == "1":
                self.test_model()
            elif choice == "2":
                if self.federated_nodes:
                    self.do_federated_learning(self.federated_nodes[0])
                else:
                    print("❌ No nodes connected")
            elif choice == "3":
                self.train_multiple_rounds(5)
            elif choice == "4":
                if self.federated_nodes:
                    self.compare_models(self.federated_nodes[0])
                else:
                    print("❌ No nodes connected")
            elif choice == "5":
                self.discover_federated_nodes()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

if __name__ == "__main__":
    print("🚀 SIMPLE FEDERATED LEARNING - SENDER")
    print("="*50)
    print("Goal: Train neural network across two laptops")
    print("Method: Share gradients, not data")
    print("="*50)
    
    sender = SimpleFederatedSender()
    
    # Test initial model
    sender.test_model()
    
    # Find nodes
    if sender.discover_federated_nodes():
        sender.interactive_mode()
    else:
        print("😢 No federated learning nodes found")