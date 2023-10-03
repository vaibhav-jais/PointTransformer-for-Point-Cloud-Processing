import matplotlib.pyplot as plt

precision_PointNet2 = [0.03905678, 0.63003311, 0.10234748, 0.0901514, 0.01064509, 0.95329254]
recall_PointNet2 = [0.52694888, 0.72161103, 0.19304602, 0.47831966, 0.00434669, 0.70277698]

# Precision and recall values for model 2
precision_PointTransformer = [0.06551532, 0.56291876, 0.1989269, 0.06408055, 0.00899013, 0.97603766]
recall_PointTransformer = [0.51890395, 0.87088409, 0.44887116, 0.64094527, 0.40537251, 0.65423979]

# Class labels
class_labels = ["Other", "Moving_Vehicle", "Static_Vehicle", "Pedestrian", "Bike", "Background"]

# Create a scatter plot for model 1
#plt.scatter(precision_PointNet2, recall_PointNet2, label="PointNet2")

# Create a scatter plot for model 2
#plt.scatter(precision_PointTransformer, recall_PointTransformer, label="PointTransformer")

# Add labels for each point
#for i, label in enumerate(class_labels):
#    plt.annotate(label, (precision_PointNet2[i], recall_PointNet2[i]))
#    plt.annotate(label, (precision_PointTransformer[i], recall_PointTransformer[i]))

markers = ['^', 'P', '+', 'x', 'X', '.']
for i in range(len(class_labels)):
    color_PointNet2 = 'r'
    color_PointTransformer = 'g'
    plt.scatter(precision_PointNet2[i], recall_PointNet2[i], label=f'PointNet2 - {class_labels[i]}', marker=markers[i], color=color_PointNet2)
    plt.scatter(precision_PointNet2[i], recall_PointTransformer[i], label=f'PointTransformer - {class_labels[i]}', marker=markers[i], color=color_PointTransformer)

# Add labels and legend
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Precision-Recall Scatter Plot for Model Comparison')

# Show the plot
plt.show()