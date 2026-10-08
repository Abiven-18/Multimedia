import json
import matplotlib.pyplot as plt
import os

with open('output/benchmark_results.json', 'r') as f:
    data = json.load(f)

labels = ['Dense Optical Flow', 'PhysioSync-AR']
fps = [data['dense_optical_flow']['analysis_fps'], data['physiosync_ar']['analysis_fps']]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, fps, color=['#d9534f', '#5cb85c'])
plt.title('Throughput Comparison (FPS)')
plt.ylabel('Frames per Second (higher is better)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, str(yval), ha='center', fontweight='bold')
plt.savefig('output/throughput_chart.png')
plt.close()

f1 = [data['dense_optical_flow']['f1'], data['physiosync_ar']['f1']]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, f1, color=['#d9534f', '#5cb85c'])
plt.title('Keyframe Selection Accuracy (F1 Score)')
plt.ylabel('F1 Score (higher is better)')
plt.ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, str(yval), ha='center', fontweight='bold')
plt.savefig('output/f1_score_chart.png')
plt.close()

ops = [data['dense_optical_flow']['op_proxy'], data['physiosync_ar']['op_proxy']]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, ops, color=['#d9534f', '#5cb85c'])
plt.yscale('log')
plt.title('Compute Operations Proxy (Log Scale)')
plt.ylabel('Number of Operations (lower is better)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,}', ha='center', va='bottom', fontweight='bold')
plt.savefig('output/compute_ops_chart.png')
plt.close()

print('Charts saved.')
