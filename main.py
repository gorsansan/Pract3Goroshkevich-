import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import operator
from tkinter import filedialog as fd
import tkinter.messagebox as mb
from os.path import dirname as up
from skimage.measure import block_reduce
from skimage import io, measure, transform,metrics

def gradi (file):
    # Размер окна и величина смещения
    ksize = 3
    dx = 1
    dy = 1

    # Вычисление градиента
    x_gradient = cv2.Sobel(file, cv2.CV_32F, dx, 0, ksize=ksize)
    y_gradient = cv2.Sobel(file, cv2.CV_32F, 0, dy, ksize=ksize)

    # Вычисление абсолютного значения градиента
    abs_x_gradient = cv2.convertScaleAbs(x_gradient)
    abs_y_gradient = cv2.convertScaleAbs(y_gradient)

    # Вычисление итогового градиента
    tot_gradient = cv2.addWeighted(abs_x_gradient, 0.5, abs_y_gradient, 0.5, 0)

    SummGrad = []
    for i in range(0, len(tot_gradient), 1):
        SummGrad.append(round(sum(tot_gradient[i]) / len(tot_gradient[i]), 1))
    return SummGrad


def DCT(file):
    # Применение двумерного дискретного косинусного преобразования (DCT)
    dct = cv2.dct(np.float32(file))
    return dct

def DFT(file):

    # Применение двумерного дискретного преобразования Фурье (DFT)
    dft = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Сдвиг нулевых частот в центр
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

def Hist(file):
    # reading the input image
    # computing the histogram of the blue channel of the image
    histog = cv2.calcHist([file], [0], None, [256], [0, 256])
    return histog

def Scale(file):
    return block_reduce(file, block_size=(2, 2), func=np.mean)


def plot_grafs(num_e):
    stat_dct = []
    stat_dft = []
    stat_scale = []
    stat_histog = []
    stat_gradi = []
    delta_k_h = 200
    delta_k_g = 150
    test_img_a = []
    test_histog= []
    test_gradi = []
    test_dft = []
    test_dct = []
    test_scale = []
    etal_img_a = []
    etal_histog = []
    etal_gradi = []
    etal_dft = []
    etal_dct = []
    etal_scale = []
    for i in range(1,11,1):
        sum_histog = 0
        sum_gradi = 0
        sum_sim_dft = 0
        sum_sim_dct=0
        sum_sim_scale=0
        for j in range(1,num_e+1,1):
            res_histog = 0
            res_gradi = 0
            fln_etal = f"s{i}/{j}.pgm"
            etal_img = cv2.imread(fln_etal, cv2.IMREAD_GRAYSCALE)
            etal_img_a.append(etal_img)
            etal_histog.append(Hist(etal_img))
            etal_gradi.append(gradi(etal_img))
            etal_dft.append(DFT(etal_img))
            etal_dct.append(DCT(etal_img))
            etal_scale.append(Scale(etal_img))

            for k in range(num_e+1, 11, 1):
                fln_t=f"S{i}/{k}.pgm"
                t_img = cv2.imread(fln_t, cv2.IMREAD_GRAYSCALE)
                test_img_a.append(t_img)
                test_histog.append(Hist(t_img))
                test_gradi.append(gradi(t_img))
                test_dft.append(DFT(t_img))
                test_dct.append(DCT(t_img))
                test_scale.append(Scale(t_img))
                in_e_h, e_m_h = max(enumerate(etal_histog[j-1+num_e*(i-1)]), key=operator.itemgetter(1))
                in_e_g, e_m_g = max(enumerate(etal_gradi[j-1+num_e*(i-1)]), key=operator.itemgetter(1))
                t_max_h = test_histog[k - num_e-1+(10-num_e)*(i-1)][in_e_h]
                t_max_g = test_gradi[k - num_e-1+(10-num_e)*(i-1)][in_e_g]
                delt_h = abs(e_m_h - t_max_h)
                delt_g = abs(e_m_g - t_max_g)
                if (delt_h < delta_k_h):
                    res_histog +=1
                if (delt_g < delta_k_g):
                    res_gradi += 1
                mean_mag_e = np.mean(etal_dft[j-1+num_e*(i-1)])
                mean_mag_t = np.mean(test_dft[k - num_e-1+(10-num_e)*(i-1)])
                similarity_percent_dft = mean_mag_t / mean_mag_e
                if(similarity_percent_dft>1):
                    similarity_percent_dft = 2 - similarity_percent_dft
                sum_sim_dft +=similarity_percent_dft
                linalg_norm_e = np.linalg.norm(etal_dct[j-1+num_e*(i-1)])
                linalg_norm_t = np.linalg.norm(test_dct[k - num_e-1+(10-num_e)*(i-1)])
                similarity_percent_dct = linalg_norm_t/linalg_norm_e
                if (similarity_percent_dct > 1):
                    similarity_percent_dct = 2 - similarity_percent_dct
                sum_sim_dct += similarity_percent_dct
                ssim = np.corrcoef(etal_scale[j-1+num_e*(i-1)].flatten(), test_scale[k - num_e-1+(10-num_e)*(i-1)].flatten())[0][1]
                if (ssim > 1):
                    ssim = 2 - ssim
                sum_sim_scale +=ssim

            sum_histog+=res_histog
            sum_gradi += res_gradi
        stat_histog.append(sum_histog/((10-num_e)*num_e))
        stat_gradi.append(sum_gradi / ((10 - num_e) * num_e))
        stat_dft.append(sum_sim_dft/((10 - num_e) * num_e))
        stat_dct.append(sum_sim_dct/((10 - num_e) * num_e))
        stat_scale.append(sum_sim_scale / ((10 - num_e) * num_e))

    fig1, ((ax_1, ax_2, ax_3, ax_4, ax_5, ax_6),(ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(2, 6)
    fig3, (axHistog, axGradi,axDFT,axDCT, axScale) = plt.subplots(1, 5)
    plt.ion()
    ax_1.set_title('Тестовая')
    i_a = ax_1.imshow(test_img_a[0])
    ax_2.set_title('Гистограмма')
    histog_a, = ax_2.plot(test_histog[0], color="b")
    ax_3.set_title('DFT')
    dft_a = ax_3.imshow(test_dft[0], cmap='gray', vmin=0, vmax=255)
    ax_4.set_title('DCT')
    dct_a = ax_4.imshow(np.abs(test_dct[0]), vmin=0, vmax=255)
    x = np.arange(len(test_gradi[0]))
    ax_5.set_title('Градиент')
    gradi_a, = ax_5.plot(x, test_gradi[0], color="b")
    ax_6.set_title('Scale')
    scale_a = ax_6.imshow(test_scale[0])

    ax1.set_title('Оригинал')
    i_a_e = ax1.imshow(etal_img_a[0])
    ax2.set_title('Гистограмма')
    histog_a_e, = ax2.plot(etal_histog[0], color="b")
    ax3.set_title('DFT')
    dft_a_e = ax3.imshow(etal_dft[0], cmap='gray', vmin=0, vmax=255)
    ax4.set_title('DCT')
    dct_a_e = ax4.imshow(np.abs(etal_dct[0]), vmin=0, vmax=255)
    x_e = np.arange(len(etal_gradi[0]))
    ax5.set_title('Градиент')
    gradi_a_e, = ax5.plot(x_e, etal_gradi[0], color="b")
    ax6.set_title('Scale')
    scale_a_e = ax6.imshow(etal_scale[0])
    x_r_gradi = np.arange(len(stat_gradi))
    x_r_histog = np.arange(len(stat_histog))
    x_r_dft = np.arange(len(stat_dft))
    x_r_dct = np.arange(len(stat_dct))
    x_r_scale = np.arange(len(stat_scale))
    axHistog.plot(x_r_histog, stat_histog, color="b")
    axHistog.set_title('Hist')
    axGradi.plot(x_r_gradi, stat_gradi, color="b")
    axGradi.set_title('Градиент')
    axDFT.plot(x_r_dft, stat_dft, color="b")
    axDFT.set_title('DFT')
    axDCT.plot(x_r_dct, stat_dct, color="b")
    axDCT.set_title('DVT')
    axScale.plot(x_r_scale, stat_scale, color="b")
    axScale.set_title('Scale')
    fig1.show()
    fig3.show()

    for t in range(0, 10, 1):
        for p in range(0+num_e*t, num_e*t+num_e, 1):
            i_a_e.set_data(etal_img_a[p])
            histog_a_e.set_ydata(etal_histog[p])
            dft_a_e.set_data(etal_dft[p])
            dct_a_e.set_data(etal_dct[p])
            gradi_a_e.set_ydata(etal_gradi[p])
            scale_a_e.set_data(etal_scale[p])
            for m in range((0+p*(10-num_e)), (10-num_e)*(p+1), 1):
                i_a.set_data(test_img_a[m])
                histog_a.set_ydata(test_histog[m])
                dft_a.set_data(test_dft[m])
                dct_a.set_data(test_dct[m])
                gradi_a.set_ydata(test_gradi[m])
                scale_a.set_data(test_scale[m])
                fig1.canvas.draw()
                fig1.canvas.flush_events()



def get_num_etalons():
    num_etalons = num_etalons_entry.get()
    if num_etalons.isdigit() and int(num_etalons) > 0:
        plot_grafs(int(num_etalons))
    else:
        tk.showerror("Ошибка", "Введите целое положительное число")

def choose_test():
    filename1 = fd.askopenfilename()
    filename2 = fd.askopenfilename()
    plot_grafs_choosen(filename1, filename2)

def show_res(text):
    msg = text
    mb.showinfo("Результат", msg)

def plot_grafs_choosen(filename1, filename2):
    delta_k_h = 100
    delta_k_g = 70
    res_h = 0
    res_g = 0
    d = up(up(filename1))
    d = d + "/"
    filename1 = filename1.replace(d,'')
    etal_img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    etal_hist = Hist(etal_img)
    etal_grad = gradi(etal_img)
    etal_dft = DFT(etal_img)
    etal_dct = DCT(etal_img)
    etal_scale = Scale(etal_img)
    d = up(up(filename2))
    d = d + "/"
    filename2 = filename2.replace(d,'')
    test_img = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    test_hist = Hist(test_img)
    test_grad = gradi(test_img)
    test_dft = DFT(test_img)
    test_dct = DCT(test_img)
    test_scale = Scale(test_img)
    test_or_img = plt.imread(filename2, cv2.IMREAD_GRAYSCALE)
    in_e_h, e_m_h = max(enumerate(etal_hist), key=operator.itemgetter(1))
    in_e_g, e_m_g = max(enumerate(etal_grad), key=operator.itemgetter(1))
    t_max_h = test_hist[in_e_h]
    t_max_g = test_grad[in_e_g]
    delt_c = abs(e_m_h - t_max_h)
    delt_g = abs(e_m_g - t_max_g)
    if(delt_c < delta_k_h):
         res_h +=1
    if (delt_g < delta_k_g):
        res_g += 1
    plt.subplot(3, 6, 13)
    plt.imshow(etal_img)
    plt.title("Эталон")
    plt.subplot(3, 6, 14)
    plt.plot(etal_hist, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 15)
    plt.imshow(etal_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(etal_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 17)
    x = np.arange(len(etal_grad))
    plt.plot(x, etal_grad, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 18)
    plt.imshow(etal_scale)
    plt.title("Scale")

    plt.subplot(3, 6, 1)
    plt.imshow(test_or_img)
    plt.title("Тестовая")
    plt.subplot(3, 6, 2)
    plt.plot(test_hist, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 3)
    plt.imshow(test_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(test_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 5)
    x = np.arange(len(test_grad))
    plt.plot(x, test_grad, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 6)
    plt.imshow(test_scale)
    plt.title("Scale")
    if (res_g != 0 and res_h != 0):
        show_res("Совпадает")
    else:
        show_res("Не совпадает")
    plt.show()

# Создаем главное окно
root = tk.Tk()

# Создаем метку и поле для ввода количества эталонов
num_etalons_label = tk.Label(root, text="Количество эталонов:")
num_etalons_label.pack()
num_etalons_entry = tk.Entry(root)
num_etalons_entry.pack()

# Создаем кнопку для подтверждения ввода
plot_button = tk.Button(root, text="Построить графики", command=get_num_etalons)
plot_button.pack()

plot_button = tk.Button(root, text="Произвести произвольную выборку", command=choose_test)
plot_button.pack()

# Запускаем главный цикл обработки событий
root.mainloop()

