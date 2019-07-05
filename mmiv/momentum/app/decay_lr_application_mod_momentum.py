import tensorflow as tf

from niftynet.application.segmentation_application import \
    SegmentationApplication
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE, NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.layer.loss_segmentation import LossFunction

SUPPORTED_INPUT = set(['image', 'label', 'weight'])


class DecayLearningRateApplication(SegmentationApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, is_training):
        SegmentationApplication.__init__(
            self, net_param, action_param, is_training)
        tf.logging.info('starting decay learning segmentation application')
        self.learning_rate = None
        self.momentum = None
        self.mom = action_param.mom
        self.current_lr = action_param.lr
        self.rate = action_param.max_iter//10
        if self.action_param.validation_every_n > 0:
            raise NotImplementedError("validation process is not implemented "
                                      "in this demo.")
        self.prec_loss = 10.0
        self.loss = None
        self.curent_loss = None


    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, self.is_training)

            with tf.name_scope('Optimiser'):
                self.learning_rate = tf.placeholder(tf.float64, shape=[])
                self.momentum = tf.placeholder(tf.float64, shape=[])

                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                   learning_rate=self.learning_rate, momentum=self.momentum)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))

            self.loss = data_loss
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                self.loss = data_loss + reg_loss
            grads = self.optimiser.compute_gradients(self.loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=self.loss, name='dice_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.learning_rate, name='lr',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.momentum, name='mom',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            SegmentationApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)

    def set_iteration_update(self, iteration_message):
        """
        This function will be called by the application engine at each
        iteration.
        """
        self.current_loss = self.loss.eval()
        current_iter = iteration_message.current_iter
        if iteration_message.is_training:
            if current_iter > 0 and current_iter % self.rate == 0:
                modif = (self.prec_loss - self.current_loss)/self.prec_loss

                self.current_lr = self.current_lr * (1 - modif)
                self.mom = self.mom * (1 + modif)

                self.prec_loss = self.current_loss
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True
        iteration_message.data_feed_dict[self.learning_rate] = self.current_lr
        iteration_message.data_feed_dict[self.momentum] = self.mom
