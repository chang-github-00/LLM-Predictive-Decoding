# import numpy as np



# def multi_iteration_sampling(N, T, p):
#     return 1 - (1 - (1-p)**T)**(N*k)

# def mpc_sampling(N, T, p,k):
#     result = 1
#     for m in range(T):
#         length = min(k, T-m)
#         result *= 1 - (1 - (1-p)**length)**(N)
        
#     return result

# p = 0.2
# T = 10
# k = 1
# N = np.arange(1, 100, 1)

# print(multi_iteration_sampling(N, T, p) - mpc_sampling(N, T, p, k))

import numpy as np
import random
import matplotlib.pyplot as plt

def sample_n(T, A):
    results = []
    for i in range(T):
        results.append(random.choice(A))
    return results
    
def mpc_sampling(T,A, k):
    
    result = []
    for i in range(T):
        best = 0
        best_future = []
        for kk in range(k):
            all_futures = []
            for j in range(i, T):
                all_futures.append(random.choice(A))
            if sum(all_futures) > best:
                best_future = all_futures
                best = sum(all_futures)
        result.append(best_future[0])
            
            
    return result


def sample_n_prior(T, A, prior):
    results = []
    for i in range(T):
        results.append(random.choices(A, prior)[0])
    return results

def mpc_sampling_prior(T,A, k, prior):
    
    result = []
    for i in range(T):
        best = 0
        best_future = []
        for kk in range(k):
            all_futures = []
            for j in range(i, T):
                all_futures.append(random.choices(A, prior)[0])
            if sum(all_futures) > best:
                best_future = all_futures
                best = sum(all_futures)
        result.append(best_future[0])
            
            
    return result

# T = 5
# A = [1, 2, 3, 4]


# plot_success_1 = []
# plot_success_2 = []
# plot_T = [3,4,5,6,7,8,9,10,11,12,13,14,15]



# N = [1000]#range(4, 1000, 20)


# for T in plot_T:
#     for N_ in N:
#         success_1 = 0
#         for attempt in range(500):
#             sample = N_//2
#             all_samples = [sample_n(T, A) for i in range(sample)]

#             result = []
#             found = False
#             for i in range(sample):
#                 if all_samples[i] == [4]*T:
#                     found = True
#                     break
#             success_1 += found

#         success_2 = 0
#         for attempt in range(500):
#             result = mpc_sampling(T, A, N_//len(A)) 
#             # print(result)

#             success_2 += (result == [4]*T)
#         plot_success_1.append(success_1/500)
#         plot_success_2.append(success_2/500)
#         print(T, N_, success_1/500, success_2/500)
        
# plt.figure(figsize=(5,5))
# plt.plot(plot_T, plot_success_1, label='Naive Iterative Sampling', marker='o')
# plt.plot(plot_T, plot_success_2, label='Model Predictive Control', marker='o')
# plt.xlabel('Problem Scale (T)')
# plt.ylabel('Success Rate With 1000 Samples')

# plt.legend()
# plt.savefig('sample_efficiency_wrt_problem_scale.pdf', format='pdf',bbox_inches='tight')


T = 5
A = [1, 2, 3, 4]



plot_T = [6]



N = range(100, 2000, 200)#range(4, 1000, 20)
tau = [1, 5, 10]

plt.figure(figsize=(5,5))

for j, t in enumerate(tau):
    plot_success_1 = []
    plot_success_2 = []
    for T in plot_T:
        for N_ in N:
            prior = np.exp(np.array(A)/t)/np.sum(np.exp(np.array(A)/t))
            success_1 = 0
            for attempt in range(500):
                sample = N_//2
                all_samples = [sample_n_prior(T, A,prior) for i in range(sample)]

                result = []
                found = False
                for i in range(sample):
                    if all_samples[i] == [4]*T:
                        found = True
                        break
                success_1 += found

            success_2 = 0
            for attempt in range(500):
                result = mpc_sampling_prior(T, A, N_//len(A), prior) 
                # print(result)

                success_2 += (result == [4]*T)
            plot_success_1.append(success_1/500)
            plot_success_2.append(success_2/500)
            print(T, N_, success_1/500, success_2/500)
        

    
    plt.plot(N, plot_success_2, label='Model Predictive Control τ='+str(t), marker='o', color='blue', alpha=0.3*(3-j))
    plt.plot(N, plot_success_1, label='Iterative Sampling τ='+str(t), marker='o', color='red', alpha=0.3*(3-j))
    
plt.xlabel('Sampling Number (N)')
plt.ylabel('Accuracy')

plt.legend(fontsize=6)
plt.savefig('sample_efficiency_wrt_sample_num.pdf', format='pdf',bbox_inches='tight')
