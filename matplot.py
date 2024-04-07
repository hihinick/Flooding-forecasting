#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
class mat_boxplot():
    def __init__(self, selected_col, diff_col, compute_col, num_in_onepit, pic_title):
        self.selected_col = selected_col
        self.diff_col = diff_col
        self.compute_col = compute_col
        self.num_in_onepit = num_in_onepit
        self.title = pic_title 
    def Sub_Boxplot(self,add_compare, add_compare_lables):
        fig7, ax7 = plt.subplots()
        ax7.set_title(self.title)
        ax7.boxplot(add_compare,labels=(add_compare_lables))
        plt.xlabel(self.diff_col)
        plt.ylabel(self.compute_col)
        plt.show()
    def multiple_Boxplot(self):
        diff = list(set(self.selected_col[self.diff_col]))
        add_compare,add_compare_lables = [[] for i in range(2)]
        n = 1
        round_ = 1
        quotient = len(diff)/ self.num_in_onepit
        remainder = len(diff)% self.num_in_onepit
        for one in diff:
            one_of_diff = self.selected_col[self.selected_col[self.diff_col] == one]
            plot = one_of_diff[self.compute_col].values
            add_compare.append(plot)
            add_compare_lables.append(one)
            if n % self.num_in_onepit == 0:   
                self.Sub_Boxplot(add_compare, add_compare_lables)
                add_compare,add_compare_lables = [[] for i in range(2)]
                round_ += 1
            elif  (round_ >quotient) & (n % self.num_in_onepit ==remainder):
                self.Sub_Boxplot(add_compare, add_compare_lables)
            n+=1

