import socket
import json
import threading
import time
import random
import math

class SimpleMLReceiver:
    def __init__(self):
        self.port = 8080
        self.discovery_port = 8079
        self.sock = None
        self.discovery_sock = None
        self.my_ip = self.get_my_ip()
        
        # Simple Neural Network (no external libraries needed!)
        self.weights = [random.uniform(-1, 1) for _ in range(4)]  # 4 weights
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Training data (simple math patterns)
        self.local_data = self.generate_training_data()
        self.training_history = []
        
        print(f"🧠 Initialized simple neural network with random weights")
        print(f"   Weights: {[round(w, 3) for w in self.weights]}")
        print(f"   Bias: {round(self.bias, 3)}")
    
    def get_my_ip(self):
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))
            ip = temp_sock.getsockname()[0]
            temp_sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def generate_training_data(self):
        """Generate simple training data - learn to predict sum of inputs"""
        data = []
        print("📊 Generating training data...")
        
        for _ in range(100):
            # Create input: [a, b, c, d]
            inputs = [random.uniform(0, 10) for _ in range(4)]
            # Target: sum of inputs (simple pattern to learn)
            target = sum(inputs)
            
            data.append({
                'inputs': inputs,
                'target': target
            })
        
        print(f"✅ Generated {len(data)} training examples")
        print(f"   Example: inputs={[round(x, 2) for x in data[0]['inputs']]} → target={round(data[0]['target'], 2)}")
        return data
    
    def sigmoid(self, x):
        """Simple activation function"""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))  # Prevent overflow
    
    def forward_pass(self, inputs):
        """Neural network forward pass"""
        # Simple: output = sigmoid(sum(weights * inputs) + bias)
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        output = self.sigmoid(weighted_sum)
        return output * 50  # Scale output to reasonable range
    
    def calculate_gradients(self, inputs, target, prediction):
        """Calculate gradients for backpropagation"""
        error = prediction - target
        
        # Simple gradient calculation
        gradients = {
            'weights': [error * inp * 0.01 for inp in inputs],  # Simplified gradient
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
        
        # Train on random batch
        batch = random.sample(self.local_data, min(batch_size, len(self.local_data)))
        
        for example in batch:
            inputs = example['inputs']
            target = example['target']
            
            # Forward pass
            prediction = self.forward_pass(inputs)
            
            # Calculate loss
            loss = (prediction - target) ** 2
            total_loss += loss
            
            # Calculate gradients
            gradients = self.calculate_gradients(inputs, target, prediction)
            
            # Accumulate gradients
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
        
        print("🔍 Testing model:")
        for i, example in enumerate(test_data):
            prediction = self.forward_pass(example['inputs'])
            error = abs(prediction - example['target'])
            total_error += error
            
            print(f"   Test {i+1}: inputs={[round(x, 1) for x in example['inputs']]} "
                  f"→ predicted={prediction:.2f}, actual={example['target']:.2f}, error={error:.2f}")
        
        avg_error = total_error / len(test_data)
        print(f"   Average error: {avg_error:.3f}")
        return avg_error
    
    def start_discovery_service(self):
        """Let other laptops find us"""
        self.discovery_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_sock.bind(('0.0.0.0', self.discovery_port))
        
        print(f"🔍 ML Discovery service running at {self.my_ip}")
        
        while True:
            try:
                message, sender_address = self.discovery_sock.recvfrom(1024)
                
                if message.decode('utf-8') == "FIND_ML_NODES":
                    response = {
                        'type': 'ml_node_info',
                        'ip': self.my_ip,
                        'port': self.port,
                        'data_size': len(self.local_data),
                        'current_weights': [round(w, 3) for w in self.weights],
                        'current_bias': round(self.bias, 3)
                    }
                    self.discovery_sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                    print(f"📡 Told {sender_address} about our ML node")
                    
            except:
                break
    
    def handle_training_message(self, data, sender_address):
        """Handle training-related messages"""
        msg_type = data.get('type')
        
        if msg_type == 'gradient_share':
            # Receive gradients from other node
            received_gradients = data['gradients']
            sender_loss = data['loss']
            
            print(f"📨 Received gradients from {sender_address}")
            print(f"   Their loss: {sender_loss:.3f}")
            
            # Train our local batch
            local_gradients, local_loss = self.train_local_batch()
            
            # Average the gradients (Federated Learning!)
            combined_gradients = {
                'weights': [
                    (local_gradients['weights'][i] + received_gradients['weights'][i]) / 2
                    for i in range(len(local_gradients['weights']))
                ],
                'bias': (local_gradients['bias'] + received_gradients['bias']) / 2
            }
            
            # Apply combined gradients
            self.apply_gradients(combined_gradients)
            
            print(f"🧠 Applied combined gradients!")
            print(f"   Local loss: {local_loss:.3f}, Remote loss: {sender_loss:.3f}")
            print(f"   New weights: {[round(w, 3) for w in self.weights]}")
            
            # Send confirmation back
            response = {
                'type': 'gradient_ack',
                'our_loss': local_loss,
                'combined_applied': True
            }
            self.sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
            
            # Test model after update
            self.test_model()
            
        elif msg_type == 'model_test_request':
            # Send our current model performance
            avg_error = self.test_model()
            response = {
                'type': 'model_performance',
                'average_error': avg_error,
                'weights': [round(w, 3) for w in self.weights],
                'bias': round(self.bias, 3),
                'data_size': len(self.local_data)
            }
            self.sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
    
    def start_receiver(self):
        """Main receiver loop"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        
        print(f"📱 ML Receiver listening on {self.my_ip}:{self.port}")
        print("🧠 Ready for distributed machine learning...")
        
        while True:
            try:
                message, sender_address = self.sock.recvfrom(8192)
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    
                    if isinstance(data, dict):
                        self.handle_training_message(data, sender_address)
                    else:
                        print(f"📨 Unknown message from {sender_address}")
                        
                except json.JSONDecodeError:
                    text = message.decode('utf-8')
                    print(f"📨 Plain text: {text}")
                    
            except KeyboardInterrupt:
                print("\n👋 ML Receiver shutting down...")
                print(f"🧠 Final model state:")
                print(f"   Weights: {[round(w, 3) for w in self.weights]}")
                print(f"   Bias: {round(self.bias, 3)}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if self.sock:
            self.sock.close()
        if self.discovery_sock:
            self.discovery_sock.close()
    
    def run(self):
        # Test initial model
        print("\n🔍 Testing initial untrained model:")
        self.test_model()
        
        # Start discovery service in background
        discovery_thread = threading.Thread(target=self.start_discovery_service)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start main receiver
        self.start_receiver()

if __name__ == "__main__":
    print("🧠 ML RECEIVER - Distributed Neural Network Training")
    print("="*60)
    print("Features: Simple Neural Network + Federated Learning + Gradient Sharing")
    print("="*60)
    
    receiver = SimpleMLReceiver()
    receiver.run()