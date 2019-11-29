# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:35:12 2019

@author: Liou Shing Tzou (108064512)
"""
import PrintResult
import prime

prime_bits = 256
plaintext_bits = 224
iter_time = 200

if __name__ == '__main__':
    
    PrintResult.Simulation1(prime.generate_prime_number, iter_time, prime_bits)
    PrintResult.Simulation2()
    PrintResult.Simulation3()
    
    
    
    
    



    
    


    