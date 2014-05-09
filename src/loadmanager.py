import logging
import numpy as np
# To manage distribution of analysis between nodes
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------
# Helper functions.

# Used by make_ring_schedule().
def gcd(n, m): # greatest common divisor
    n, m = max(n, m), min(n,m)
    while m != 0:
        n, m = m, n%m

    return n


class Loadmanager(object):

    def __init__(self,trajectory_lengths, trajectory_filepaths, mpi_size,my_rank):
        
        # ----------------------------------------------------------------
        # Divide trajectories between nodes.
        # Get trajectory data.
        # Read.
        self.trajectory_lengths = trajectory_lengths
        self.trajectory_filepaths = trajectory_filepaths
        self.my_rank = my_rank
        self.mpi_size = mpi_size

    
    def do_partition(self):
        # Offests and Ranges.
        trajectory_ID_offsets = np.insert(np.cumsum(self.trajectory_lengths)[:-1],0,0)
        trajectory_ID_ranges = zip(trajectory_ID_offsets, np.cumsum(self.trajectory_lengths))
        
        # Divide work.
        costs = self.trajectory_lengths
        items = zip(self.trajectory_filepaths, self.trajectory_lengths,
                    trajectory_ID_offsets, trajectory_ID_ranges)
        cost_sums, transposed_partitions = self._divide_work(costs, items)
        # partitions :: [[(a0,b0,..)]]
        transpose = lambda xs: zip(*xs)
        partitions = map(transpose, transposed_partitions)
        # transposed_partitions :: [([a0, a1, ...], [b0, b1, ...], ..)] # (conceptually)
        # Record the range of frames assigned to each node.
        # (Lower bound is inclusive, upper bound is exclusive.)
        # Only valid if contiguous_divide_work groups the trajectories in contiguous ranges.
        # Note that when striding != 1 this is not a contiguous range of frames that a node stores.
        (parts_trajectory_filepaths, parts_trajectory_lengths, \
            parts_trajectory_ID_offsets, parts_trajectory_ID_ranges) = \
            map(list, zip(*partitions))
            
        self.frame_globalID_distribution = list(parts_trajectory_ID_ranges)
        self.partitions = partitions
        
    def get_myworkshare(self):
        
        logger.debug("Fetching workshare for %s: %s",self.my_rank, self.partitions[self.my_rank])
        return self.partitions[self.my_rank]
        
        
        

    def _divide_work(self, costs, items):
        """
        Returns a partitioning of the items which attempts to minimize the maximum total cost
        of all partitions.
    
        Expected Types:
        - pieces      :: Integer
        - costs       :: [Integer]
        - items       :: [a]
        - return_type :: [[a]]
        """
        
        logger.debug("Calculating load divisions")
        
        parts = [ [] for x in xrange(self.mpi_size) ]
        sums  = [ 0 for x in xrange(self.mpi_size) ]
    
        sorted_cost_item = reversed(sorted(zip(costs, items), key = lambda x: x[0]))
    
        for cost, item in sorted_cost_item:
            min_i = sums.index(min(sums))
            parts[min_i].append(item)
            sums[min_i] += cost
    
        return sums, parts




    def make_half_ring_schedule(self):
        return [ (i, i+(self.mpi_size/2)) for i in xrange(self.mpi_size/2) ]
    

    
    def make_ring_schedule(self, k, size=None): # need to rename function and 'k'
        """
        Expected Types:
        - k           :: Integer
        - size        :: Integer
        - return_type :: Schdule [[(NodeID Integer, NodeID Integer)]]
    
        We have n nodes, arranged in a ring. So, the node 'after' the n'th node is the 1st node.
        Let the nodes be numbered 0, 1, ..., n-1.
    
        We want each node to send a message to the node 'k' places ahead of it in the ring.
        So, node 'i' sends a message to node 'i+k mod n'.
    
        So then, you can see that each node sends exactly one thing and recieves exactly one thing.
    
    
        This function uses a simple method to generate a schedule
        where pairs of nodes can communicate simultaneously.
    
        A consequence of this type of messaging is that the nodes form one or more cycles.
        Given that we use blocking communication, the rings will have either a number of members
        that is either even or odd.
        In the even case, we need only two 'messaging rounds', but we need three for the odd case.
        """
        if size is None: size = self.mpi_size
        
        number_of_groups = gcd(size, k)
        group_size = size / number_of_groups
        
        EVEN_PARITY = 0; ODD_PARITY = 1
        if (group_size % 2) == 0:
            group_size_parity = EVEN_PARITY
            # number_of_rounds = 2
        else:
            group_size_parity = ODD_PARITY
            # number_of_rounds = 3
    
    
        schedule = []
    
        for g in xrange(number_of_groups):
            for i in xrange(group_size/2):
                schedule.append(( (2*i*k  )   + g,  (2*i + 1)*k + g))
                schedule.append(( (2*i + 1)*k + g,  (2*i + 2)*k + g))
    
            if group_size_parity == ODD_PARITY:
                schedule.append((  -1*k       + g,                g))
                    
        # Take modulus with respect to the number of nodes.
        mod_pairs = lambda pair: (pair[0] % size, pair[1] % size)
        schedule = map(mod_pairs, schedule)
        return schedule


    def run_schedule(self,schedule, my_rank, mpi_send_lambda, mpi_recv_lambda):
        
        for node_pair in schedule:
            send_rank, dest_rank = node_pair
            if my_rank == send_rank:
                mpi_send_lambda(dest_rank)
                logger.info("rank %s sending to %s ",my_rank, dest_rank )
    
            elif my_rank == dest_rank:
                mpi_recv_lambda(send_rank)
                logger.info("rank %s recieving from %s ",my_rank, send_rank )
    

    def find_node_of_frame(self,frameID):
    
        for node_id, frame_ranges in enumerate(self.frame_globalID_distribution):
            for frame_range in frame_ranges:
                low, high = frame_range
                if low <= frameID < high:
                    return node_id
        return None

               
     