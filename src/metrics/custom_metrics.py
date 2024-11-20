import tensorflow as tf

class WeightedRecall(tf.keras.metrics.Metric):
    def __init__(self, name='weighted_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_recall = self.add_weight(
            name='total_recall', 
            initializer='zeros'
            )
        self.total_weight = self.add_weight(
            name='total_weight', 
            initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot encoded labels to class indices
        y_true = tf.argmax(y_true, axis=-1)  # Collapse one-hot encoding
        y_true = tf.reshape(y_true, [-1])    # Ensure it's rank 1
    
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.reshape(y_pred, [-1])    # Ensure consistency with y_true
    
        # Get unique classes and their weights
        unique_classes, _ = tf.unique(y_true)
        num_classes = tf.shape(unique_classes)[0]
    
        # Initialize accumulators for recalls and weights
        recalls = tf.TensorArray(dtype=tf.float32, size=num_classes)
        weights = tf.TensorArray(dtype=tf.float32, size=num_classes)
    
        # Loop over unique classes
        for i in tf.range(num_classes):
            class_id = unique_classes[i]
            class_mask = tf.equal(y_true, class_id)
    
            # Mask predictions and ground truths
            y_true_class = tf.boolean_mask(y_true, class_mask)
            y_pred_class = tf.boolean_mask(y_pred, class_mask)
    
            # Calculate recall for this class
            recall = tf.keras.metrics.Recall()
            recall.update_state(y_true_class, y_pred_class)
            class_recall = recall.result()
    
            # Store recall and weight
            recalls = recalls.write(i, class_recall)
            weights = weights.write(i, tf.reduce_sum(tf.cast(class_mask, tf.float32)))
    
        # Compute weighted recall
        recalls = recalls.stack()
        weights = weights.stack()
        total_weight = tf.reduce_sum(weights)
        weighted_recall = tf.reduce_sum(recalls * weights) / total_weight
    
        # Update metric state
        self.total_recall.assign_add(weighted_recall)
        self.total_weight.assign_add(total_weight)


    
    def result(self):
        return self.total_recall / self.total_weight
    
    def reset_states(self):
        self.total_recall.assign(0)
        self.total_weight.assign(0)
