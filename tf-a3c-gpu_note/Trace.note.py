# =============enqueue_op=============
0. Where use? 
~a3c.py:75~
queue = tf.FIFOQueue(capacity=NUM_THREADS*10,
                        dtypes=[tf.float32,tf.float32,tf.float32],)
qr = tf.train.QueueRunner(queue, [g_agent.enqueue_op(queue) for g_agent in group_agents])
tf.train.queue_runner.add_queue_runner(qr)
loss = queue.dequeue()


1. 
~async_agent.py:63~
def enqueue_op(self,queue) :




# ============= Where pre process?=============



#  ============= Where pre process?=============

def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format


Know NCHW & strides


#========= target_values ===========#
0. Where use? 
~async_agent.py:125~
target_values = np.stack(values,axis=0).astype(np.float32)

1. target_values before & after code

>>>         vs = self.value_func(np.stack(self.states,axis=0))
            states = []
            actions = []
            values = []

            # print('len(sras)={}',len(sras))
            # 'len(sras)= AGENT_PER_THREADS = 12'
            for i,sra in enumerate(sras) :
                # print('sra.shape = {}'.format(np.shape(sra)  ))
                # sra.shape = (5, 3)
                # 5-> 'UNROLL_STEP', 3 -> (self.states[i],reward,action)
                if i in done_envs :
                    vs[i] = 0.
                    self.states[i] = None

                for s,r,a in sra[::-1] :
>>>                 vs[i] = r + self.discount_factor*vs[i]
                    states.append(s)
                    actions.append(a)
>>>                 values.append(vs[i])

            states = np.stack(states,axis=0)
            actions = np.stack(actions,axis=0).astype(np.int32)
>>>         target_values = np.stack(values,axis=0).astype(np.float32)

>>>         policy_loss, entropy_loss, value_loss = self.ac.update(states,actions,target_values)
            self.ac.sync()

2. self.value_func ? 

2.1. from __init__
~async_agent.py:31~
class A3CGroupAgent():
    def __init__(self,envs,actor_critic,unroll_step,discount_factor,
                 seed=None,image_size=(84,84),frames_for_state=4) :
        ......
        self.value_func = actor_critic.get_value

2.2. actor_critic.get_value
~network.py:108~
class ActorCritic():
    ...
    def get_value(self, s):
        return self.sess.run(self.value,
                             feed_dict={self.state: s})


#========= HistoryBuffer ===========#

0. Where Use?
~async_agent.py:50~
class A3CGroupAgent():
    def __init__(self,envs,actor_critic,unroll_step,discount_factor,
                 seed=None,image_size=(84,84),frames_for_state=4) :
                 ...
        self.hist_bufs = [HistoryBuffer(self.preprocess,self.observe_shape,frames_for_state) for _ in envs]
        
~async_agent.py:85~
    def enqueue_op(self,queue) :
        def _func():
            # Initialize states, if the game is done at the last iteration
            for i,_ in enumerate(self.envs) :
                if self.states[i] is None :
                    self.episode_rewards[i].append( self.episode_reward[i] )
                    self.episode_reward[i] = 0.

>>>>                self.hist_bufs[i].clear()
                    o = self.envs[i].reset()
>>>>                self.states[i] = self.hist_bufs[i].add(o)

            ....
            for step in range(self.unroll_step) :
                actions = self.pick_action(np.stack(self.states,axis=0))

                for i,(env,action) in enumerate(zip(self.envs,actions)) :
                    if( i in done_envs ) :
                        continue

                    o, reward, done, _ = env.step(action)
                    self.episode_reward[i] += reward

                    reward = max(-1,min(1,reward)) #reward clippint -1 to 1
                    sras[i].append((self.states[i],reward,action))
>>>>                self.states[i] = self.hist_bufs[i].add(o)


1. HistoryBuffer detail

class HistoryBuffer():
    def __init__(self,preprocess_fn,image_shape,frames_for_state) :
        self.buf = deque(maxlen=frames_for_state)
        self.preprocess_fn = preprocess_fn
        self.image_shape = image_shape
        self.clear()

    def clear(self) :
        for i in range(self.buf.maxlen):
            self.buf.append(np.zeros(self.image_shape,np.float32))

    def add(self,o) :
        #assert( list(o.shape) == self.image_shape ),'%s, %s'%(o.shape,self.image_shape)
        self.buf.append(self.preprocess_fn(o))
        state = np.concatenate([img for img in self.buf], axis=2)
        return state
#========= Trains_var ===========#

0. Where use? 
~network.py:71~

master:train_vars
<tf.Variable 'master/conv1/b:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'master/conv1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>
<tf.Variable 'master/conv2/b:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'master/conv2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>
<tf.Variable 'master/linear-policy/b:0' shape=(4,) dtype=float32_ref>
<tf.Variable 'master/linear-policy/w:0' shape=(256, 4) dtype=float32_ref>
<tf.Variable 'master/linear-value/b:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'master/linear-value/w:0' shape=(256, 1) dtype=float32_ref>
<tf.Variable 'master/linear1/b:0' shape=(256,) dtype=float32_ref>
<tf.Variable 'master/linear1/w:0' shape=(7744, 256) dtype=float32_ref>
<network.ActorCritic instance at 0x7f864b974a28> = ActorCritic(4,device_name=/gpu:0,                             learning_rate=Tensor("PolynomialDecay:0", shape=(), dtype=float32),decay=0.99,grad_clip=0.1,                             entropy_beta=0.01)
('self.state', <tf.Tensor 'Placeholder_1:0' shape=(?, 84, 84, 4) dtype=float32>)
Thread00:train_vars
<tf.Variable 'Thread00/conv1/b:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Thread00/conv1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>
<tf.Variable 'Thread00/conv2/b:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'Thread00/conv2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>
<tf.Variable 'Thread00/linear-policy/b:0' shape=(4,) dtype=float32_ref>
<tf.Variable 'Thread00/linear-policy/w:0' shape=(256, 4) dtype=float32_ref>
<tf.Variable 'Thread00/linear-value/b:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'Thread00/linear-value/w:0' shape=(256, 1) dtype=float32_ref>
<tf.Variable 'Thread00/linear1/b:0' shape=(256,) dtype=float32_ref>
<tf.Variable 'Thread00/linear1/w:0' shape=(7744, 256) dtype=float32_ref>
('self.state', <tf.Tensor 'Placeholder_4:0' shape=(?, 84, 84, 4) dtype=float32>)
Thread01:train_vars
<tf.Variable 'Thread01/conv1/b:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Thread01/conv1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>
<tf.Variable 'Thread01/conv2/b:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'Thread01/conv2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>
<tf.Variable 'Thread01/linear-policy/b:0' shape=(4,) dtype=float32_ref>
<tf.Variable 'Thread01/linear-policy/w:0' shape=(256, 4) dtype=float32_ref>
<tf.Variable 'Thread01/linear-value/b:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'Thread01/linear-value/w:0' shape=(256, 1) dtype=float32_ref>
<tf.Variable 'Thread01/linear1/b:0' shape=(256,) dtype=float32_ref>
<tf.Variable 'Thread01/linear1/w:0' shape=(7744, 256) dtype=float32_ref>




#========= Network ===========#

    with tf.device(device_name) :
            self.state = tf.placeholder(tf.float32,[None]+state_shape)
            print('self.state', self.state)

            block, self.scope  = ActorCritic._build_shared_block(self.state,scope_name)
            self.policy, self.log_softmax_policy = ActorCritic._build_policy(block,nA,scope_name)
            self.value = ActorCritic._build_value(block,scope_name)

            self.train_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name), key=lambda v:v.name)
            # print('self.train_vars -> ')
            # print(self.train_vars)
            # print('{}:train_vars'.format(self.scope.name))


            print('self.policy={}, self.log_softmax_policy={}'.format(self.policy, self.log_softmax_policy) )
            print('self.policy={}'.format(self.value) )
            

            # for v in self.train_vars:
            #     print(v)

            if( master is not None ) :
                self.sync_op= self._sync_op(master)
                self.action = tf.placeholder(tf.int32,[None,])  
                #self.action=Tensor("Placeholder_2:0", shape=(?,), dtype=int32, device=/device:GPU:0)
                
                self.target_value = tf.placeholder(tf.float32,[None,])
                # self.target_value=Tensor("Placeholder_3:0", shape=(?,), dtype=float32, device=/device:GPU:0)

            
                
                advantage = self.target_value - self.value   
                # shape=(?,) = shape=(?,) - shape=(?,)
                # Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0) = Tensor("Placeholder_3:0", shape=(?,), dtype=float32, device=/device:GPU:0) - Tensor("Thread00_2/Squeeze:0", shape=(?,), dtype=float32, device=/device:GPU:0)
                
                entropy = tf.reduce_sum(-1. * self.policy * self.log_softmax_policy,axis=1)
                # shape=(?,) = tf.reduce_sum(-1. * shape=(?, 4) * shape=(?, 4),axis=1)
                # Tensor("Sum:0", shape=(?,), dtype=float32, device=/device:GPU:0) = tf.reduce_sum(-1. * Tensor("Thread00_1/Softmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0) * Tensor("Thread00_1/LogSoftmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0),axis=1)

        
                log_p_s_a = tf.reduce_sum(self.log_softmax_policy * tf.one_hot(self.action,nA),axis=1)
                # shape=(?,) = tf.reduce_sum(shape=(?, 4) * tf.one_hot(shape=(?,), 4),axis=1)
                # Tensor("Sum_1:0", shape=(?,), dtype=float32, device=/device:GPU:0) = tf.reduce_sum(Tensor("Thread00_1/LogSoftmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0) * tf.one_hot(Tensor("Placeholder_2:0", shape=(?,), dtype=int32, device=/device:GPU:0)),axis=1)


                advantage   = self.target_value - self.value
                # shape=(?,)= shape=(?,)        - shape=(?,)
                # Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0) = Tensor("Placeholder_3:0", shape=(?,), dtype=float32, device=/device:GPU:0) - Tensor("Thread00_2/Squeeze:0", shape=(?,), dtype=float32, device=/device:GPU:0)

                self.policy_loss = tf.reduce_mean(tf.stop_gradient(advantage)   *log_p_s_a)
                #    shape=()    = tf.reduce_mean(tf.stop_gradient(shape=(?,))*shape=(?,))
                # Tensor("Mean:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(tf.stop_gradient(Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0))*Tensor("Sum_1:0", shape=(?,), dtype=float32, device=/device:GPU:0))
                self.entropy_loss = tf.reduce_mean(entropy)
                #        shape=() = tf.reduce_mean(shape=(?,) )
                # Tensor("Mean_1:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(Tensor("Sum:0", shape=(?,), dtype=float32, device=/device:GPU:0))

                self.value_loss = tf.reduce_mean(advantage**2)
                #      shape=() = tf.reduce_mean( shape=(?,) ** 2)
                # Tensor("Mean_2:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0)**2)

                loss = -self.policy_loss - entropy_beta* self.entropy_loss + self.value_loss
                # shape=() = -shape=()   - 0.01        * shape=()          + shape=()       
                # Tensor("add:0", shape=(), dtype=float32, device=/device:GPU:0) = -Tensor("Mean:0", shape=(), dtype=float32, device=/device:GPU:0) - 0.01* Tensor("Mean_1:0", shape=(), dtype=float32, device=/device:GPU:0) + Tensor("Mean_2:0", shape=(), dtype=float32, device=/device:GPU:0)


                self.gradients = tf.gradients(loss,self.train_vars)
                # [<tf.Tensor 'gradients/Thread00/*]  =   tf.gradients( shape=(), [<tf.Variable 'Thread00/*]
                # [<tf.Tensor 'gradients/Thread00/BiasAdd_grad/BiasAddGrad:0' shape=(32,) dtype=float32>, <tf.Tensor 'gradients/Thread00/Conv2D_grad/Conv2DBackpropFilter:0' shape=(8, 8, 4, 32) dtype=float32>, <tf.Tensor 'gradients/Thread00/BiasAdd_1_grad/BiasAddGrad:0' shape=(64,) dtype=float32>, <tf.Tensor 'gradients/Thread00/Conv2D_1_grad/Conv2DBackpropFilter:0' shape=(4, 4, 32, 64) dtype=float32>, <tf.Tensor 'gradients/Thread00_1/add_grad/Reshape_1:0' shape=(4,) dtype=float32>, <tf.Tensor 'gradients/Thread00_1/MatMul_grad/MatMul_1:0' shape=(256, 4) dtype=float32>, <tf.Tensor 'gradients/Thread00_2/add_grad/Reshape_1:0' shape=(1,) dtype=float32>, <tf.Tensor 'gradients/Thread00_2/MatMul_grad/MatMul_1:0' shape=(256, 1) dtype=float32>, <tf.Tensor 'gradients/Thread00/add_grad/Reshape_1:0' shape=(256,) dtype=float32>, <tf.Tensor 'gradients/Thread00/MatMul_grad/MatMul_1:0' shape=(7744, 256) dtype=float32>] 
                #   = tf.gradients(Tensor("add:0", shape=(), dtype=float32, device=/device:GPU:0),self.[<tf.Variable 'Thread00/conv1/b:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Thread00/conv1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>, <tf.Variable 'Thread00/conv2/b:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Thread00/conv2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'Thread00/linear-policy/b:0' shape=(4,) dtype=float32_ref>, <tf.Variable 'Thread00/linear-policy/w:0' shape=(256, 4) dtype=float32_ref>, <tf.Variable 'Thread00/linear-value/b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Thread00/linear-value/w:0' shape=(256, 1) dtype=float32_ref>, <tf.Variable 'Thread00/linear1/b:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'Thread00/linear1/w:0' shape=(7744, 256) dtype=float32_ref>])
                #         ('self.state', <tf.Tensor 'Placeholder_4:0' shape=(?, 84, 84, 4) dtype=float32>)


                clipped_gs = [tf.clip_by_average_norm(g,grad_clip) for g in self.gradients]
                self.train_op = master.optimizer.apply_gradients(zip(clipped_gs,master.train_vars))
                
            else :
                #self.optimizer = tf.train.AdamOptimizer(learning_rate,beta1=BETA)
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=decay,use_locking=True)



self.action=Tensor("Placeholder_2:0", shape=(?,), dtype=int32, device=/device:GPU:0)
self.target_value=Tensor("Placeholder_3:0", shape=(?,), dtype=float32, device=/device:GPU:0)
advantage = self.target_value - self.value
Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0) = Tensor("Placeholder_3:0", shape=(?,), dtype=float32, device=/device:GPU:0) - Tensor("Thread00_2/Squeeze:0", shape=(?,), dtype=float32, device=/device:GPU:0)
entropy = tf.reduce_sum(-1. * self.policy * self.log_softmax_policy,axis=1)
Tensor("Sum:0", shape=(?,), dtype=float32, device=/device:GPU:0) = tf.reduce_sum(-1. * Tensor("Thread00_1/Softmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0) * Tensor("Thread00_1/LogSoftmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0),axis=1)
log_p_s_a = tf.reduce_sum(self.log_softmax_policy * tf.one_hot(self.action,nA),axis=1)
       Tensor("Sum_1:0", shape=(?,), dtype=float32, device=/device:GPU:0) = tf.reduce_sum(Tensor("Thread00_1/LogSoftmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0) * tf.one_hot(Tensor("Placeholder_2:0", shape=(?,), dtype=int32, device=/device:GPU:0),4),axis=1)
self.policy_loss = tf.reduce_mean(tf.stop_gradient(advantage)*log_p_s_a)
              Tensor("Mean:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(tf.stop_gradient(Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0))*Tensor("Sum_1:0", shape=(?,), dtype=float32, device=/device:GPU:0))
self.entropy_loss = tf.reduce_mean(entropy)
               Tensor("Mean_1:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(Tensor("Sum:0", shape=(?,), dtype=float32, device=/device:GPU:0))
self.value_loss = tf.reduce_mean(advantage**2)
             Tensor("Mean_2:0", shape=(), dtype=float32, device=/device:GPU:0) = tf.reduce_mean(Tensor("sub:0", shape=(?,), dtype=float32, device=/device:GPU:0)**2)
loss = -self.policy_loss - entropy_beta* self.entropy_loss + self.value_loss
  Tensor("add:0", shape=(), dtype=float32, device=/device:GPU:0) = -Tensor("Mean:0", shape=(), dtype=float32, device=/device:GPU:0) - 0.01* Tensor("Mean_1:0", shape=(), dtype=float32, device=/device:GPU:0) + Tensor("Mean_2:0", shape=(), dtype=float32, device=/device:GPU:0)
self.gradients = tf.gradients(loss,self.train_vars)
            [<tf.Tensor 'gradients/Thread00/BiasAdd_grad/BiasAddGrad:0' shape=(32,) dtype=float32>, <tf.Tensor 'gradients/Thread00/Conv2D_grad/Conv2DBackpropFilter:0' shape=(8, 8, 4, 32) dtype=float32>, <tf.Tensor 'gradients/Thread00/BiasAdd_1_grad/BiasAddGrad:0' shape=(64,) dtype=float32>, <tf.Tensor 'gradients/Thread00/Conv2D_1_grad/Conv2DBackpropFilter:0' shape=(4, 4, 32, 64) dtype=float32>, <tf.Tensor 'gradients/Thread00_1/add_grad/Reshape_1:0' shape=(4,) dtype=float32>, <tf.Tensor 'gradients/Thread00_1/MatMul_grad/MatMul_1:0' shape=(256, 4) dtype=float32>, <tf.Tensor 'gradients/Thread00_2/add_grad/Reshape_1:0' shape=(1,) dtype=float32>, <tf.Tensor 'gradients/Thread00_2/MatMul_grad/MatMul_1:0' shape=(256, 1) dtype=float32>, <tf.Tensor 'gradients/Thread00/add_grad/Reshape_1:0' shape=(256,) dtype=float32>, <tf.Tensor 'gradients/Thread00/MatMul_grad/MatMul_1:0' shape=(7744, 256) dtype=float32>] = tf.gradients(Tensor("add:0", shape=(), dtype=float32, device=/device:GPU:0),self.[<tf.Variable 'Thread00/conv1/b:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Thread00/conv1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>, <tf.Variable 'Thread00/conv2/b:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Thread00/conv2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'Thread00/linear-policy/b:0' shape=(4,) dtype=float32_ref>, <tf.Variable 'Thread00/linear-policy/w:0' shape=(256, 4) dtype=float32_ref>, <tf.Variable 'Thread00/linear-value/b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Thread00/linear-value/w:0' shape=(256, 1) dtype=float32_ref>, <tf.Variable 'Thread00/linear1/b:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'Thread00/linear1/w:0' shape=(7744, 256) dtype=float32_ref>])
('self.state', <tf.Tensor 'Placeholder_4:0' shape=(?, 84, 84, 4) dtype=float32>)
self.policy=Tensor("Thread01_1/Softmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0), self.log_softmax_policy=Tensor("Thread01_1/LogSoftmax:0", shape=(?, 4), dtype=float32, device=/device:GPU:0)
self.policy=Tensor("Thread01_2/Squeeze:0", shape=(?,), dtype=float32, device=/device:GPU:0)
self.sync_op=name: "group_deps_1"