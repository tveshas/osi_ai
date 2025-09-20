import socket
import json
import time
import random
import math

class SimpleMLSender:
    def __init__(self):
        self.discovery_port = 8079
        self.receivers = []
        
        # Simple Neural Network (same architecture as receiver)
        self.weights = [random.uniform(-1, 1) for _ in range(4)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Our own training data (different from receiver)
        self.local_data = self.generate_training_data()
        self.training_rounds = 0
        
        print(f"🧠 Initialized ML sender with random weights")
        print(f"   Weights: {[round(w, 3) for w in self.weights]}")
        print(f"   Bias: {round(self.bias, 3)}")
    
    def generate_training_data(self):
        """Generate different training data than receiver"""
        data = []
        print("📊 Generating our training data...")
        
        for _ in range(100):
            # Slightly different pattern - helps with generalization
            inputs = [random.uniform(2, 8) for _ in range(4)]  # Different range
            target = sum(inputs) + random.uniform(-1, 1)  # Add some noise
            
            data.append({
                'inputs': inputs,
                'target': target
            })
        
        print(f"✅ Generated {len(data)} training examples")
        print(f"   Example: inputs={[round(x, 2) for x in data[0]['inputs']]} → target={round(data[0]['target'], 2)}")
        return data
    
    def sigmoid(self, x):
        """Simple activation function"""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def forward_pass(self, inputs):
        """Neural network forward pass"""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        output = self.sigmoid(weighted_sum)
        return output * 50  # Scale output
    
    def calculate_gradients(self, inputs, target, prediction):
        """Calculate gradients for backpropagation"""
        error = prediction - target
        
        gradients = {
            'weights': [error * inp * 0.01 for inp in inputs],
            'bias': error * 0.01
        }
        
        return gradients
    
    def apply_gradients(self, gradients):
        """Update weights using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients['weights'][i]
        
        self.bias -= self.learning_rate * gradients['bias']
    
    def train_local_batch(self, batch_size=10):
        """Train on local data and return gradients"""
        print(f"🎯 Training on {batch_size} local examples...")
        
        total_gradients = {'weights': [0] * 4, 'bias': 0}
        total_loss = 0
        
        batch = random.sample(self.local_data, min(batch_size, len(self.local_data)))
        
        for example in batch:
            inputs = example['inputs']
            target = example['target']
            
            prediction = self.forward_pass(inputs)
            loss = (prediction - target) ** 2
            total_loss += loss
            
            gradients = self.calculate_gradients(inputs, target, prediction)
            
            for i in range(len(gradients['weights'])):
                total_gradients['weights'][i] += gradients['weights'][i]
            total_gradients['bias'] += gradients['bias']
        
        # Average gradients
        for i in range(len(total_gradients['weights'])):
            total_gradients['weights'][i] /= len(batch)
        total_gradients['bias'] /= len(batch)
        
        avg_loss = total_loss / len(batch)
        print(f"   Local loss: {avg_loss:.3f}")
        return total_gradients, avg_loss
    
    def test_model(self):
        """Test current model performance"""
        test_data = random.sample(self.local_data, 5)
        total_error = 0
        
        print("🔍 Testing our model:")
        for i, example in enumerate(test_data):
            prediction = self.forward_pass(example['inputs'])
            error = abs(prediction - example['target'])
            total_error += error
            
            print(f"   Test {i+1}: inputs={[round(x, 1) for x in example['inputs']]} "
                  f"→ predicted={prediction:.2f}, actual={example['target']:.2f}, error={error:.2f}")
        
        avg_error = total_error / len(test_data)
        print(f"   Our average error: {avg_error:.3f}")
        return avg_error
    
    def find_ml_nodes(self):
        """Find ML receivers on network"""
        print("🔍 Searching for ML training nodes...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(3.0)
        
        broadcast_message = "FIND_ML_NODES"
        networks = ["192.168.1.255", "192.168.0.255", "10.0.0.255", "127.0.0.1"]
        
        for network in networks:
            try:
                sock.sendto(broadcast_message.encode('utf-8'), (network, self.discovery_port))
            except:
                continue
        
        found_any = False
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            try:
                response, address = sock.recvfrom(4096)
                
                try:
                    node_info = json.loads(response.decode('utf-8'))
                    
                    if node_info.get('type') == 'ml_node_info':
                        node_ip = node_info['ip']
                        node_port = node_info['port']
                        data_size = node_info.get('data_size', 0)
                        weights = node_info.get('current_weights', [])
                        bias = node_info.get('current_bias', 0)
                        
                        if (node_ip, node_port) not in [(r['ip'], r['port']) for r in self.receivers]:
                            self.receivers.append({
                                'ip': node_ip,
                                'port': node_port,
                                'data_size': data_size,
                                'weights': weights,
                                'bias': bias
                            })
                            
                            print(f"✅ Found ML node at {node_ip}:{node_port}")
                            print(f"   📊 Their data size: {data_size}")
                            print(f"   🧠 Their weights: {weights}")
                            found_any = True
                
                except json.JSONDecodeError:
                    continue
                        
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
        
        if not found_any:
            print("❌ No ML nodes found automatically.")
            manual_ip = input("🔧 Enter ML receiver IP manually (or press Enter to skip): ")
            if manual_ip.strip():
                self.receivers.append({
                    'ip': manual_ip.strip(),
                    'port': 8080,
                    'data_size': 0,
                    'weights': [],
                    'bias': 0
                })
        
        return len(self.receivers) > 0
    
    def send_gradients_to_node(self, receiver):
        """Send our gradients to a receiver for federated learning"""
        print(f"\n🚀 Starting federated learning with {receiver['ip']}...")
        
        # Calculate our gradients
        local_gradients, local_loss = self.train_local_batch()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10.0)
        
        # Send gradients
        message = {
            'type': 'gradient_share',
            'gradients': local_gradients,
            'loss': local_loss,
            'round': self.training_rounds
        }
        
        try:
            print(f"📤 Sending gradients to {receiver['ip']}...")
            print(f"   Our loss: {local_loss:.3f}")
            print(f"   Gradient weights: {[round(g, 4) for g in local_gradients['weights']]}")
            
            sock.sendto(json.dumps(message).encode('utf-8'), (receiver['ip'], receiver['port']))
            
            # Wait for acknowledgment
            response, addr = sock.recvfrom(4096)
            ack_data = json.loads(response.decode('utf-8'))
            
            if ack_data.get('type') == 'gradient_ack':
                remote_loss = ack_data['our_loss']
                print(f"✅ Federated learning completed!")
                print(f"   Remote loss: {remote_loss:.3f}")
                print(f"   Combined gradients applied on both sides")
                
                self.training_rounds += 1
                return True
                
        except socket.timeout:
            print("⏰ Training request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        sock.close()
        return False
    
    def request_model_comparison(self, receiver):
        """Compare our model with receiver's model"""
        print(f"\n📊 Requesting model comparison from {receiver['ip']}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        request = {'type': 'model_test_request'}
        
        try:
            sock.sendto(json.dumps(request).encode('utf-8'), (receiver['ip'], receiver['port']))
            
            response, addr = sock.recvfrom(4096)
            data = json.loads(response.decode('utf-8'))
            
            if data.get('type') == 'model_performance':
                their_error = data['average_error']
                their_weights = data['weights']
                their_bias = data['bias']
                their_data_size = data['data_size']
                
                # Test our own model
                our_error = self.test_model()
                
                print(f"\n📈 MODEL COMPARISON:")
                print(f"   Our model error: {our_error:.3f}")
                print(f"   Their model error: {their_error:.3f}")
                print(f"   Performance difference: {abs(our_error - their_error):.3f}")
                
                print(f"\n🧠 WEIGHT COMPARISON:")
                print(f"   Our weights: {[round(w, 3) for w in self.weights]}")
                print(f"   Their weights: {their_weights}")
                
                print(f"\n📊 DATA COMPARISON:")
                print(f"   Our data size: {len(self.local_data)}")
                print(f"   Their data size: {their_data_size}")
                
                return data
                
        except socket.timeout:
            print("⏰ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        sock.close()
        return {}
    
    def train_multiple_rounds(self, rounds=5):
        """Perform multiple rounds of federated learning"""
        if not self.receivers:
            print("❌ No ML nodes available for training")
            return
        
        print(f"\n🚀 Starting {rounds} rounds of federated learning...")
        
        initial_error = self.test_model()
        
        for round_num in range(rounds):
            print(f"\n📍 ROUND {round_num + 1}/{rounds}")
            print("="*40)
            
            for receiver in self.receivers:
                success = self.send_gradients_to_node(receiver)
                if success:
                    print(f"✅ Round {round_num + 1} completed with {receiver['ip']}")
                else:
                    print(f"❌ Round {round_num + 1} failed with {receiver['ip']}")
                
                time.sleep(1)  # Small delay between rounds
        
        final_error = self.test_model()
        
        print(f"\n🎯 TRAINING SUMMARY:")
        print(f"   Initial error: {initial_error:.3f}")
        print(f"   Final error: {final_error:.3f}")
        print(f"   Improvement: {initial_error - final_error:.3f}")
        print(f"   Training rounds completed: {self.training_rounds}")
    
    def interactive_mode(self):
        """Interactive menu for ML operations"""
        while True:
            print("\n" + "="*60)
            print("🧠 ML SENDER - Distributed Neural Network Training")
            print(f"🎯 Training rounds completed: {self.training_rounds}")
            print(f"📊 Local training data: {len(self.local_data)} examples")
            print(f"🌐 Connected ML nodes: {len(self.receivers)}")
            print("="*60)
            print("1️⃣  Test our current model")
            print("2️⃣  Send gradients (1 round of federated learning)")
            print("3️⃣  Train multiple rounds (5 rounds)")
            print("4️⃣  Compare models with other nodes")
            print("5️⃣  Find ML nodes again")
            print("6️⃣  Show weight evolution")
            print("7️⃣  Quit")
            
            choice = input("Choose (1-7): ")
            
            if choice == "1":
                self.test_model()
            elif choice == "2":
                if self.receivers:
                    self.send_gradients_to_node(self.receivers[0])
                else:
                    print("❌ No ML nodes connected")
            elif choice == "3":
                self.train_multiple_rounds(5)
            elif choice == "4":
                for receiver in self.receivers:
                    self.request_model_comparison(receiver)
            elif choice == "5":
                self.receivers = []
                self.find_ml_nodes()
            elif choice == "6":
                self.show_weight_evolution()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    def show_weight_evolution(self):
        """Show how weights have evolved"""
        print(f"\n🧠 CURRENT MODEL STATE:")
        print(f"   Weights: {[round(w, 3) for w in self.weights]}")
        print(f"   Bias: {round(self.bias, 3)}")
        print(f"   Training rounds: {self.training_rounds}")
        print(f"   Learning rate: {self.learning_rate}")

if __name__ == "__main__":
    print("🚀 ML SENDER - Distributed Neural Network Training")
    print("="*60)
    print("Features: Federated Learning + Gradient Sharing + Model Comparison")
    print("="*60)
    
    sender = SimpleMLSender()
    
    # Test initial model
    print("\n🔍 Testing initial untrained model:")
    sender.test_model()
    
    if sender.find_ml_nodes():
        sender.interactive_mode()
    else:
        print("😢 Could not find any ML training nodes")
        print("💡 Make sure ml_receiver.py is running on another laptop")