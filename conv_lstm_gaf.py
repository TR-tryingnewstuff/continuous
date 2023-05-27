import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch


class ConvLSTM(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ConvLSTM, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        self.image_15 = tf.keras.layers.Input(shape=original_space['image_15'].shape, name="image_15")
        self.image_d = tf.keras.layers.Input(shape=original_space['image_d'].shape, name="image_d")
        
        self.time = tf.keras.layers.Input(shape=original_space['time'].shape, name="time")
        
       # self.weekday = tf.keras.layers.Input(shape=original_space['weekday'].shape, name="weekday")

        # Concatenating the inputs;
        # One can pass different parts of the state to different networks before concatenation.
        
        conv1_15 = tf.keras.layers.ConvLSTM2D(8, 2, (2, 2), activation='tanh', padding='same')(self.image_15)
         
        drop1_15 = tf.keras.layers.Dropout(0.1)(conv1_15)   
        conv2_15 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2), activation='tanh', padding='same')(drop1_15)       
        maxpool1_15 = tf.keras.layers.MaxPool2D( (2, 2))(conv2_15)

        drop2_15 = tf.keras.layers.Dropout(0.1)(maxpool1_15)
        conv3_15 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2), activation='tanh', padding='same')(drop2_15)
    
        
        conv4_15 = tf.keras.layers.Conv2D(8, (2, 2), (2, 2))(conv3_15)
      
        
        conv_flat_15 = tf.keras.layers.Flatten()(conv4_15)
        conv_out_15 = tf.keras.layers.Dense(1, activation='tanh')(conv_flat_15)
        
        # ----------- DAILY ---------------------
        conv1_d = tf.keras.layers.ConvLSTM2D(8, 2, (2, 2), activation='tanh', padding='same')(self.image_d)
               
        drop1_d = tf.keras.layers.Dropout(0.2)(conv1_d)   
        conv2_d = tf.keras.layers.Conv2D(8, (2, 2), (2, 2), activation='tanh', padding='same')(drop1_d)       
        maxpool1_d = tf.keras.layers.MaxPool2D(( 2, 2))(conv2_d)

        drop2_d = tf.keras.layers.Dropout(0.2)(maxpool1_d)
        conv3_d = tf.keras.layers.Conv2D(8, (2, 2), (2, 2), activation='tanh', padding='same')(drop2_d)
        maxpool2_d = tf.keras.layers.MaxPool2D(( 2, 2))(conv3_d)

        
        conv_flat_d = tf.keras.layers.Flatten()(maxpool2_d)
        conv_out_d = tf.keras.layers.Dense(1, activation='tanh')(conv_flat_d)
        
        
        concatenated = tf.keras.layers.Concatenate()([conv_out_15, conv_out_d, self.time])

        # Building the dense layers
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.image_15, self.image_d, self.time], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image_15': orig_obs["image_15"], 'image_d': orig_obs["image_d"], 'time': orig_obs['time']}
        model_out, self._value_out = self.base_model(inputs) 
        
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  