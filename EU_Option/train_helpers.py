import tensorflow as tf

from EU_Option.model_subclass import MyModel



def training_loop(train_dataset, val_dataset, epochs, batch, input_dim, mode,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
                  val_acc_metric=tf.keras.metrics.MeanAbsoluteError(), activation='relu'):
    model = MyModel(input_dim=input_dim, mode=mode, activation=activation)

    losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f'############ START OF EPOCH {epoch + 1} ################')
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            grads, loss = model.get_grad_and_loss(x_batch_train, y_batch_train)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(float(loss))

            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                print(f'Seen so far: {(step + 1) * batch} samples')

        losses.append(loss)

        for val_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_logits = model.call(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print(f'Validation acc: {val_acc}')

        val_losses.append(val_acc)
    return model, losses, val_losses


def adaptive_training_loop(train_dataset, val_dataset, epochs, batch, input_dim, mode, learning_rates, lr_epochs,
                           val_acc_metric=tf.keras.metrics.MeanAbsoluteError(), activation='relu'):
    model = MyModel(input_dim=input_dim, mode=mode, activation=activation)

    losses = []
    val_losses = []

    for epoch in range(epochs):
        if epoch < lr_epochs[0]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
        elif lr_epochs[0] <= epoch < lr_epochs[1]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[1])
        elif lr_epochs[1] <= epoch < lr_epochs[2]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[2])

        print(f'############ START OF EPOCH {epoch + 1} ################')
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            grads, loss = model.get_grad_and_loss(x_batch_train, y_batch_train)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(float(loss))

            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                print(f'Seen so far: {(step + 1) * batch} samples')

        losses.append(loss)

        for val_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_logits = model.call(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print(f'Validation acc: {val_acc}')

        val_losses.append(val_acc)
    return model, losses, val_losses


def hybrid_training_loop(train_dataset, val_dataset, epochs, batch, input_dim, mode, learning_rates, lr_epochs,
                         val_acc_metric=tf.keras.metrics.MeanAbsoluteError(), activation='relu'):
    model = MyModel(input_dim=input_dim, mode=mode, activation=activation)

    losses = []
    val_losses = []

    for epoch in range(epochs):
        if epoch < lr_epochs[0]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
            model.mode = 'mse'
        elif lr_epochs[0] <= epoch < lr_epochs[1]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[1])
            model.mode = 'mse'
        elif lr_epochs[1] <= epoch < lr_epochs[2]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[2])
            model.mode = 'mse'
        elif epoch >= lr_epochs[2]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[2])
            model.mode = 'u_T'

        print(f'############ START OF EPOCH {epoch + 1} ################')
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            grads, loss = model.get_grad_and_loss(x_batch_train, y_batch_train)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(float(loss))

            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                print(f'Seen so far: {(step + 1) * batch} samples')

        losses.append(loss)

        for val_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_logits = model.call(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print(f'Validation acc: {val_acc}')

        val_losses.append(val_acc)
    return model, losses, val_losses