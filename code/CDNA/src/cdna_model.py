
def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """
        Sample batch with specified mix of ground truth and generated data points.
        e.g: the final matrix is a mix of vectors from the ground_truth (gt) and prediction (p)
            [gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10] = ground truth
            [p1, p2, p3, p4, p5, p6] = prediction
            [p1, gt2, gt3, gt4, p5, p6, gt7, gt8, gt9, gt10] = returns

        Args:
            ground_truth_x: tensor of ground-truth data point
            generated_x: tensor of generated data point
            batch_size: batch size
            num_ground_truth: number of ground-truth examples to include in batch
        Returns:
            New batch with num_ground_truth samples from ground_truth_x and the rest from generated_x
    """
    xp = chainer.cuda.get_array_module(generated_x.data)
    ground_truth_x = chainer.cuda.to_cpu(ground_truth_x)
    generated_x = chainer.cuda.to_cpu(generated_x.data)

    idx = np.arange(int(batch_size))
    np.random.shuffle(idx)
    ground_truth_idx = np.array(np.take(idx, np.arange(num_ground_truth)))
    generated_idx = np.array(np.take(idx, np.arange(num_ground_truth, int(batch_size))))

    reshaped_ground_truth_x = F.reshape(ground_truth_x, (int(batch_size), -1))
    reshaped_genetated_x = F.reshape(generated_x, (int(batch_size), -1))

    ground_truth_examps = np.take(reshaped_ground_truth_x.data, ground_truth_idx, axis=0)
    generated_examps = np.take(reshaped_genetated_x.data, generated_idx, axis=0)

    index_a = np.vstack((ground_truth_idx, np.zeros_like(ground_truth_idx)))
    index_b = np.vstack((generated_idx, np.ones_like(generated_idx)))
    ground_truth_generated_stacked = np.hstack((ground_truth_idx, generated_idx))
    ground_truth_generated_stacked_sorted = np.argsort(ground_truth_generated_stacked)
    order = np.hstack((index_a, index_b))[:, ground_truth_generated_stacked_sorted]

    stitched = []
    for i in xrange(len(order[0])):
        if order[1][i] == 0:
            pos = np.where(ground_truth_idx == i)
            stitched.append(ground_truth_examps[pos])
            continue
        else:
            pos = np.where(generated_idx == i)
            stitched.append(generated_examps[pos])
            continue
    stitched = np.array(stitched, dtype=np.float32)
    stitched = np.reshape(stitched, (ground_truth_x.shape[0], ground_truth_x.shape[1], ground_truth_x.shape[2], ground_truth_x.shape[3]))
    return xp.array(stitched)

def peak_signal_to_noise_ratio(true, pred):
    """
        Image quality metric based on maximal signal power vs. power of the noise

        Args:
            true: the ground truth image
            pred: the predicted image
        Returns:
            Peak signal to noise ratio (PSNR)
    """
    return 10.0 * F.log(1.0 / F.mean_squared_error(true, pred)) / log(10.0)


def broadcast_reshape(x, y, axis=0):
    """
        Reshape y to correspond to shape of x

        Args:
            x: the broadcasted
            y: the broadcastee
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_shape = tuple([1] * axis + list(y.shape) +
                [1] * (len(x.shape) - axis - len(y.shape)))
    y_t = F.reshape(y, y_shape)
    y_t = F.broadcast_to(y_t, x.shape)
    return y_t

def broadcasted_division(x, y, axis=0):
    """
        Apply a division x/y where y is broadcasted to x to be able to complete the operation
        
        Args:
            x: the numerator
            y: the denominator
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x / y_t

def broadcast_scale(x, y, axis=0):
    """ 
        Apply a multiplication x*y where y is broadcasted to x to be able to complete the operation

        Args:
            x: left hand operation
            y: right hand operation
            axis: where the reshape will be performed
        Resuts:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x*y_t

# =============
# Chains (chns)
# =============

class LayerNormalizationConv2D(chainer.Chain):    
    
    def __init__(self):
        super(LayerNormalizationConv2D, self).__init__()

        with self.init_scope():
            self.norm = L.LayerNormalization()

    
    """
        Apply a "layer normalization" on the result of a convolution

        Args:
            inputs: input tensor, 4D, batch x channel x height x width
        Returns:
            Output variable of shape (batch x channels x height x width)
    """
    def __call__(self, inputs):
        batch_size, channels, height, width = inputs.shape[0:4]
        inputs = F.reshape(inputs, (batch_size, -1))
        inputs = self.norm(inputs)
        inputs = F.reshape(inputs, (batch_size, channels, height, width))
        return inputs


# =============
# Models (mdls)
# =============


class BasicConvLSTMCell(chainer.Chain):
    """ Stateless convolutional LSTM, as seen in lstm_op.py from video_prediction model """

    def __init__(self, out_size=None, filter_size=5):
        super(BasicConvLSTMCell, self).__init__()

        with self.init_scope():
            # @TODO: maybe provide in channels because the concatenation
            self.conv = L.Convolution2D(4*out_size, (filter_size, filter_size), pad=filter_size/2)

        self.out_size = out_size
        self.filter_size = filter_size
        self.reset_state()

    def reset_state(self):
        self.c = None
        self.h = None

    def __call__(self, inputs, forget_bias=1.0):
        """Basic LSTM recurrent network cell, with 2D convolution connctions.

          We add forget_bias (default: 1) to the biases of the forget gate in order to
          reduce the scale of forgetting in the beginning of the training.

          It does not allow cell clipping, a projection layer, and does not
          use peep-hole connections: it is the basic baseline.

          Args:
            inputs: input Tensor, 4D, batch x channels x height x width
            forget_bias: the initial value of the forget biases.

          Returns:
             a tuple of tensors representing output and the new state.
        """ 
        # In Tensorflow: batch x height x width x channels
        # In Chainer: batch x channel x height x width
        # Create a state based on Finn's implementation
        xp = chainer.cuda.get_array_module(*inputs.data)
        if self.c is None:
            self.c = xp.zeros((inputs.shape[0], self.out_size, inputs.shape[2], inputs.shape[3]), dtype=inputs[0].data.dtype)
        if self.h is None:
            self.h = xp.zeros((inputs.shape[0], self.out_size, inputs.shape[2], inputs.shape[3]), dtype=inputs[0].data.dtype)

        #c, h = F.split_axis(state, indices_or_sections=2, axis=1)

        #inputs_h = np.concatenate((inputs, h), axis=1)
        inputs_h = F.concat((inputs, self.h), axis=1)

        # Parameters of gates are concatenated into one conv for efficiency
        #j_i_f_o = L.Convolution2D(in_channels=inputs_h.shape[1], out_channels=4*num_channels, ksize=(filter_size, filter_size), pad=filter_size/2)(inputs_h)
        j_i_f_o = self.conv(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        j, i, f, o = F.split_axis(j_i_f_o, indices_or_sections=4, axis=1)

        self.c = self.c * F.sigmoid(f + forget_bias) + F.sigmoid(i) * F.tanh(j)
        self.h = F.tanh(self.c) * F.sigmoid(o)

        #return new_h, np.concatenate((new_c, new_h), axis=1)
        #return new_h, F.concat((new_c, new_h), axis=1)
        return self.h


class StatelessCDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using CDNA
        * Because the CDNA does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """
    
    def __init__(self, num_masks):
        super(StatelessCDNA, self).__init__()

        with self.init_scope():
            self.enc7 = L.Deconvolution2D(in_channels=64, out_channels=3, ksize=(1,1), stride=1)
            self.cdna_kerns = L.Linear(in_size=None, out_size=DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks)

        self.num_masks = num_masks

    def __call__(self, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessCDNA.
            Args:
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)
        
        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

        img_height = prev_image.shape[2]
        img_width = prev_image.shape[3]

        # CDNA specific
        enc7 = self.enc7(enc6)
        enc7 = F.relu(enc7)
        transformed_list = list([F.sigmoid(enc7)])

        # CDNA specific
        # Predict kernels using linear function of last layer
        cdna_input = F.reshape(hidden5, (int(batch_size), -1))
        cdna_kerns = self.cdna_kerns(cdna_input)

        # Reshape and normalize
        # B x C x H x W => B x NUM_MASKS x 1 x H x W
        cdna_kerns = F.reshape(cdna_kerns, (int(batch_size), self.num_masks, 1, DNA_KERN_SIZE, DNA_KERN_SIZE))
        cdna_kerns = F.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = F.sum(cdna_kerns, (2, 3, 4), keepdims=True)
        cdna_kerns = broadcasted_division(cdna_kerns, norm_factor)

        # Treat the color channel dimension as the batch dimension since the same
        # transformation is applied to each color channel.
        # Treat the batch dimension as the channel dimension so that
        # F.depthwise_convolution_2d can apply a different transformation to each sample.
        cdna_kerns = F.reshape(cdna_kerns, (int(batch_size), self.num_masks, DNA_KERN_SIZE, DNA_KERN_SIZE))
        cdna_kerns = F.transpose(cdna_kerns, (1, 0, 2, 3))
        # Swap the batch and channel dimension.
        prev_image = F.transpose(prev_image, (1, 0, 2, 3))

        # Transform the image.
        transformed = F.depthwise_convolution_2d(prev_image, cdna_kerns, stride=(1, 1), pad=DNA_KERN_SIZE/2)

        # Transpose the dimensions where they belong.
        transformed = F.reshape(transformed, (color_channels, int(batch_size), self.num_masks, img_height, img_width))
        transformed = F.transpose(transformed, (2, 1, 0, 3, 4))
        transformed = F.split_axis(transformed, indices_or_sections=self.num_masks, axis=0)
        transformed = [F.squeeze(t, axis=0) for t in transformed]

        transformed_list += transformed

        return transformed_list, enc7


class Model(chainer.Chain):
    """
        This Model wrap other models like CDNA, STP or DNA.
        It calls their training and get the generated images and states, it then compute the losses and other various parameters
    """
    def __init__(self, num_masks, is_cdna=True, is_dna=False, is_stp=False, use_state=True, scheduled_sampling_k=-1, num_frame_before_prediction=2, prefix=None):
        """
            Initialize a CDNA, STP or DNA through this 'wrapper' Model
            Args:
                is_cdna: if the model should be an extension of CDNA
                is_dna: if the model should be an extension of DNA
                is_stp: if the model should be an extension of STP
                use_state: if the state should be concatenated
                scheduled_sampling_k: schedule sampling hyperparameter k
                num_frame_before_prediction: number of frame before prediction
                prefix: appended to the results to differentiate between training and validation
                learning_rate: learning rate
        """
        super(Model, self).__init__()

        with self.init_scope():
	    self.enc0 = L.Convolution2D(32, (5, 5), stride=2, pad=2)
            self.enc1 = L.Convolution2D(32, (3, 3), stride=2, pad=1)
            self.enc2 = L.Convolution2D(64, (3, 3), stride=2, pad=1)
            self.enc3 = L.Convolution2D(64, (1, 1), stride=1)

            self.enc4 = L.Deconvolution2D(128, (3, 3), stride=2, outsize=(16,16), pad=1)
            self.enc5 = L.Deconvolution2D(96, (3, 3), stride=2, outsize=(32,32), pad=1)
            self.enc6 = L.Deconvolution2D(64, (3, 3), stride=2, outsize=(64, 64), pad=1)

            self.lstm1 = BasicConvLSTMCell(32)
            self.lstm2 = BasicConvLSTMCell(32)
            self.lstm3 = BasicConvLSTMCell(64)
            self.lstm4 = BasicConvLSTMCell(64)
            self.lstm5 = BasicConvLSTMCell(128)
            self.lstm6 = BasicConvLSTMCell(64)
            self.lstm7 = BasicConvLSTMCell(32)
            
            self.norm_enc0 = LayerNormalizationConv2D()
            self.norm_enc6 = LayerNormalizationConv2D()
            self.hidden1 = LayerNormalizationConv2D()
            self.hidden2 = LayerNormalizationConv2D()
            self.hidden3 = LayerNormalizationConv2D()
            self.hidden4 = LayerNormalizationConv2D()
            self.hidden5 = LayerNormalizationConv2D()
            self.hidden6 = LayerNormalizationConv2D()
            self.hidden7 = LayerNormalizationConv2D()

            self.masks = L.Deconvolution2D(num_masks+1, (1, 1), stride=1)

            self.current_state = L.Linear(5)

            model = StatelessCDNA(num_masks)


        self.num_masks = num_masks
        self.use_state = use_state
        self.scheduled_sampling_k = scheduled_sampling_k
        self.num_frame_before_prediction = num_frame_before_prediction
        self.prefix = prefix

        self.loss = 0.0
        self.psnr_all = 0.0
        self.summaries = []
        self.conv_res = []

        # Condition ops callback
        def ops_smear(use_state):
            def ops(args):
                x = args.get("x")
                if use_state:
                    state_action = args.get("state_action")
                    batch_size = args.get("batch_size")

                    smear = F.reshape(state_action, (int(batch_size), int(state_action.shape[1]), 1, 1))
                    smear = F.tile(smear, (1, 1, int(x.shape[2]), int(x.shape[3])))
                    x = F.concat((x, smear), axis=1) # Previously axis=3 but our channel is on axis=1? ok
                return x
            return ops
        
        def ops_skip_connection(enc_idx):
            def ops(args):
                x = args.get("x")
                enc = args.get("encs")[enc_idx]
                # Skip connection (current input + target enc)
                x = F.concat((x, enc), axis=1) # Previously axis=3 but our channel is on axis=1? ok!
                return x
            return ops

        def ops_save(name):
            def ops(args):
                x = args.get("x")
                save_map = args.get("map")
                save_map[name] = x
                return x
            return ops

        def ops_get(name):
            def ops(args):
                save_map = args.get("map")
                return save_map[name]
            return ops


        # Create an executable array containing all the transformations
        self.ops = [
            [self.enc0, self.norm_enc0],
            [self.lstm1, self.hidden1, ops_save("hidden1"), self.lstm2, self.hidden2, ops_save("hidden2"), self.enc1],
            [self.lstm3, self.hidden3, ops_save("hidden3"), self.lstm4, self.hidden4, ops_save("hidden4"), self.enc2],
            [ops_smear(use_state), self.enc3],
            [self.lstm5, self.hidden5, ops_save("hidden5"), self.enc4],
            [self.lstm6, self.hidden6, ops_save("hidden6"), ops_skip_connection(1), self.enc5],
            [self.lstm7, self.hidden7, ops_save("hidden7"), ops_skip_connection(0), self.enc6, self.norm_enc6]
        ]

    def reset_state(self):
        """
            Reset the gradient of this model, but also the specific model
        """
        self.loss = 0.0
        self.psnr_all = 0.0
        self.summaries = []
        self.conv_res = []
        self.lstm1.reset_state()
        self.lstm2.reset_state()
        self.lstm3.reset_state()
        self.lstm4.reset_state()
        self.lstm5.reset_state()
        self.lstm6.reset_state()
        self.lstm7.reset_state()

    def __call__(self, x, iter_num=-1.0):
        """
            Calls the training process
            Args:
                x: an array containing an array of:
                    images: an array of Tensor of shape batch x channels x height x width
                    actions: an array of Tensor of shape batch x action
                    states: an array of Tensor of shape batch x state
                iter_num: iteration (epoch) index
            Returns:
                loss, all the peak signal to noise ratio, summaries
        """
        logger = logging.getLogger(__name__)

        # Split the images, actions and states from the input
        if len(x) > 1:
            images, actions, states = x
        else:
            images, actions, states = x[0], None, None

        batch_size, color_channels, img_height, img_width = images[0].shape[0:4]

        #img_training_set = [np.transpose(np.squeeze(img), (0, 3, 1, 2)) for img in img_training_set]
        
        # Generated robot states and images
        gen_states, gen_images = [], []
        current_state = states[0]

        # When validation/test, disable schedule sampling
        if not chainer.config.train or self.scheduled_sampling_k == -1:
            feedself = True
        else:
            # Scheduled sampling, inverse sigmoid decay
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = np.int32(
                np.round(np.float32(batch_size) * (self.scheduled_sampling_k / (self.scheduled_sampling_k + np.exp(iter_num / self.scheduled_sampling_k))))
            )
            feedself = False

        for image, action in zip(images[:-1], actions[:-1]):
            # Reuse variables after the first timestep
            reuse = bool(gen_images)

            done_warm_start = len(gen_images) > self.num_frame_before_prediction - 1
            if feedself and done_warm_start:
                # Feed in generated image
                prev_image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
                prev_image = variable.Variable(prev_image)
            else:
                # Always feed in ground_truth
                prev_image = variable.Variable(image)

            # Predicted state is always fed back in
            state_action = F.concat((action, current_state), axis=1)

            """ Execute the ops array of transformations """
            # If an ops has a name of "ops" it means it's a custom ops
            encs = []
            maps = {}
            x = prev_image
            for i in xrange(len(self.ops)):
                for j in xrange(len(self.ops[i])):
                    op = self.ops[i][j]
                    if isinstance(op, types.FunctionType):
                        # Only these values are use now in the ops callback
                        x = op({
                            "x": x, 
                            "encs": encs, 
                            "map": maps, 
                            "state_action": state_action, 
                            "batch_size": batch_size
                        })
                    else:
                        x = op(x)
                # ReLU at the end of each transformation
                x = F.relu(x)
                # At the end of j iteration = completed a enc transformation
                encs.append(x)

            # Extract the variables
            hiddens = [
                maps.get("hidden1"), maps.get("hidden2"), maps.get("hidden3"), maps.get("hidden4"),
                maps.get("hidden5"), maps.get("hidden6"), maps.get("hidden7")
            ]
            enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
            hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

            """ Specific model transformations """
            transformed, enc7 = self.model(
                encs, hiddens,
                batch_size, prev_image, self.num_masks, int(color_channels)
            )
            encs.append(enc7)

            """ Masks """
            masks = self.masks(enc6)
            masks = F.relu(masks)
            masks = F.reshape(masks, (-1, self.num_masks + 1))
            masks = F.softmax(masks)
            masks = F.reshape(masks, (int(batch_size), self.num_masks+1, int(img_height), int(img_width))) # Previously num_mask at the end, but our channels are on axis=1? ok!
            mask_list = F.split_axis(masks, indices_or_sections=self.num_masks+1, axis=1) # Previously axis=3 but our channels are on axis=1 ?

            output = broadcast_scale(prev_image, mask_list[0])
            for layer, mask in zip(transformed, mask_list[1:]):
                output += broadcast_scale(layer, mask, axis=0)
            gen_images.append(output)

            current_state = self.current_state(state_action)
            gen_states.append(current_state)

        # End of transformations
        self.conv_res = encs

        # L2 loss, PSNR for eval
        loss, psnr_all = 0.0, 0.0
        summaries = []
        for i, x, gx in zip(range(len(gen_images)), images[self.num_frame_before_prediction:], gen_images[self.num_frame_before_prediction - 1:]):
            x = variable.Variable(x)
            recon_cost = F.mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(self.prefix + '_recon_cost' + str(i) + ': ' + str(recon_cost.data))
            summaries.append(self.prefix + '_psnr' + str(i) + ': ' + str(psnr_i.data))
            loss += recon_cost
            #print(recon_cost.data)

        for i, state, gen_state in zip(range(len(gen_states)), states[self.num_frame_before_prediction:], gen_states[self.num_frame_before_prediction - 1:]):
            state = variable.Variable(state)
            state_cost = F.mean_squared_error(state, gen_state) * 1e-4
            summaries.append(self.prefix + '_state_cost' + str(i) + ': ' + str(state_cost.data))
            loss += state_cost
        
        summaries.append(self.prefix + '_psnr_all: ' + str(psnr_all.data if isinstance(psnr_all, variable.Variable) else psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - self.num_frame_before_prediction)
        summaries.append(self.prefix + '_loss: ' + str(loss.data if isinstance(loss, variable.Variable) else loss))
        
        self.summaries = summaries
        self.gen_images = gen_images
        
        return self.loss