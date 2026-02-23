import cv2
from matplotlib import pyplot as plt
import numpy as np

from dataset_analysis.ceav import CEAV


class CEAV_Exploration:


    @staticmethod
    def explore(img_path, img_total, folder_list, repo):
        # DEBUG ONLY
        luminances = []
        entropies = []
        local_entropies = []
        for i in range(img_total):
            img_holder, _ = repo.load_img(img_path, folder_list, i)
            luminance_mean, entropy, local_entropy_map = CEAV.analyse(img_holder)
            
            luminances.append(luminance_mean)
            entropies.append(entropy)
            local_entropies.append(local_entropy_map)
            if i == 1:
                break


        title = "Global entropies"
        xlabel = "mean entropy"
        ylabel = "image quantity"
        # CEAV_Exploration.histogram(entropies, title, xlabel, ylabel)

        toshow = cv2.normalize(local_entropies[0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("nada", toshow)
        # CEAV_Exploration.luminance_graph(luminances, 30)
        
    @staticmethod
    def luminance_graph(luminances, bins=30):
        plt.figure(figsize=(10, 6)) # Deixa o gráfico em um tamanho bom
        
        # O parâmetro 'bins' define em quantas "gavetas" ou colunas os dados serão divididos.
        # Você pode passar um número inteiro (ex: 30) ou uma lista de limites.
        n, bins, patches = plt.hist(luminances, bins=bins, color='royalblue', edgecolor='black', alpha=0.8)

        # Adicionando informações para ficar profissional
        plt.title('Distribuição da Luminância Média no Dataset', fontsize=14)
        plt.xlabel('Luminância Média (Ex: 0 a 255)', fontsize=12)
        plt.ylabel('Frequência (Quantidade de Imagens)', fontsize=12)
        
        # Uma linha vertical mostrando onde está a média de todo o dataset
        media_total = np.mean(luminances)
        plt.axvline(media_total, color='red', linestyle='dashed', linewidth=2, label=f'Média Geral: {media_total:.1f}')
        plt.legend()

        plt.grid(axis='y', alpha=0.5)
        plt.show()

    @staticmethod
    def histogram(data, title, xlabel, ylabel, bins=30):
        plt.figure(figsize=(10, 6)) # Deixa o gráfico em um tamanho bom
        
        # O parâmetro 'bins' define em quantas "gavetas" ou colunas os dados serão divididos.
        # Você pode passar um número inteiro (ex: 30) ou uma lista de limites.
        n, bins, patches = plt.hist(data, bins=bins, color='royalblue', edgecolor='black', alpha=0.8)

        # Adicionando informações para ficar profissional
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        # Uma linha vertical mostrando onde está a média de todo o dataset
        media_total = np.mean(data)
        plt.axvline(media_total, color='red', linestyle='dashed', linewidth=2, label=f'General average: {media_total:.1f}')
        plt.legend()

        plt.grid(axis='y', alpha=0.5)
        # plt.show()
        plt.savefig(title)

