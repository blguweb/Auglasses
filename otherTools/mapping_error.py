import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# raw 
imu_csv_path = "./rawdata/lyr2_ex.csv"
time_csv_path = "./timestamp/ex_lyr2.csv"
imu_data = pd.read_csv(imu_csv_path)
imu_data = imu_data.values
time_data = pd.read_csv(time_csv_path)
time_data = time_data.values
start_filted = 2250
end_filted = 2300

def unit_conversion(raw_data):
    raw_data[:,0] = (raw_data[:,0] * 9.8) / 16384
    raw_data[:,1] = (raw_data[:,1] * 9.8) / 16384
    raw_data[:,2] = (raw_data[:,2] * 9.8) / 16384
    raw_data[:,3] = (raw_data[:,3] * 2000) / 0x8000
    raw_data[:,4] = (raw_data[:,4] * 2000) / 0x8000
    raw_data[:,5] = (raw_data[:,5] * 2000) / 0x8000
    return raw_data

def mapping_method(A,B):
    # A -> B
    if A.shape != B.shape:
        raise ValueError("Input matrices A and B must have the same shape")
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    N = A.shape[0]
    
    A = A - np.tile(centroid_A, (N, 1))
    B = B - np.tile(centroid_B, (N, 1))
    A = np.float64(A)
    B = np.float64(B)
    H = np.dot(A.T, B)
    
    U, S, Vt = np.linalg.svd(H)
    
    R = np.dot(Vt.T, U.T)
    
    
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] = -Vt[2, :]
        R = np.dot(Vt.T, U.T)
    
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    
    return R, t

def umeyama(X, Y):
    # assert X.shape[0] == 3
    # assert Y.shape[0] == 3
    # assert X.shape[1] > 0
    # assert Y.shape[1] > 0

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)


    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T


    print(type(Xc[0,0]))
    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))
    Sxy = np.dot(Xc,Yc.T)/n


    U, D, V = np.linalg.svd(Sxy)
    V = V.T.copy()

    S = np.eye(m)

    R = np.matmul(V,U.T)
    det0 = np.linalg.det(R)

    if det0 < 0:
        print("reflection detected!")
        V_ = V.copy()
        if (abs(D) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R = np.matmul(V_, U.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R, t, c


def draw(origin,mapped):



    fig=plt.figure(figsize=(10, 15))
    imu_left_ax=fig.add_subplot(6,1,1)
    imu_left_ay=fig.add_subplot(6,1,2)
    imu_left_az=fig.add_subplot(6,1,3)

    imu_left_g_ax = fig.add_subplot(6,1,4)
    imu_left_g_ay = fig.add_subplot(6,1,5)
    imu_left_g_az = fig.add_subplot(6,1,6)

    length = origin.shape[0]
    x = np.arange(0,length)
    
    imu_left_ax.plot(x,origin[:,0],color='b')
    imu_left_ax.plot(x,mapped[:,0],color='r')

    imu_left_ay.plot(x,origin[:,1],color='b')
    imu_left_ay.plot(x,mapped[:,1],color='r')

    imu_left_az.plot(x,origin[:,2],color='b')
    imu_left_az.plot(x,mapped[:,2],color='r')

    imu_left_g_ax.plot(x,origin[:,3],color='b')
    imu_left_g_ax.plot(x,mapped[:,3],color='r')

    imu_left_g_ay.plot(x,origin[:,4],color='b')
    imu_left_g_ay.plot(x,mapped[:,4],color='r')

    imu_left_g_az.plot(x,origin[:,5],color='b')
    imu_left_g_az.plot(x,mapped[:,5],color='r')
    plt.tight_layout()
    # plt.savefig(te + '_'  + ex +".jpg")
    plt.show()


def mapping_uniform(data):
    left_data =data[start_filted:end_filted,0:6]
    right_data = data[start_filted:end_filted,6:12]
    calibration_data = data[start_filted:end_filted,12:18]
    left_data = np.float64(left_data)
    right_data = np.float64(right_data)
    calibration_data = np.float64(calibration_data)

    N = end_filted - start_filted
    print(type(left_data[0,0]))
    R_left,T_left = mapping_method(calibration_data,left_data)
    R_right,T_right = mapping_method(calibration_data,right_data)


    # R_left,T_left,c_left = umeyama(calibration_data,left_data)
    # R_right,T_right,c_right = umeyama(calibration_data,right_data)
    # print("R_left",R_left)
    # print("T_left",T_left)
    # print("c_left",c_left)

    # print("R_right",R_right)
    # print("T_right",T_right)
    # print("c_right",c_right)

    left_pred = np.dot(R_left, calibration_data.T) + np.tile(T_left.reshape(-1, 1), (1,N))
    right_pred = np.dot(R_right, calibration_data.T) + np.tile(T_right.reshape(-1, 1), (1,N))
    left_pred = np.transpose(left_pred)
    right_pred = np.transpose(right_pred)

    # (100,6)
    # 减去均值
    left_pred = left_pred - np.mean(left_pred,axis=0)
    right_pred = right_pred - np.mean(right_pred,axis=0)
    
    diff = left_pred - left_data
    err = np.sum(diff**2)
    rmse = np.sqrt(err / (N*6))
    print("left error:",rmse)

    diff = right_pred - right_data
    err = np.sum(diff**2)
    rmse = np.sqrt(err / (N*6))
    print("right error:",rmse)

    draw(right_data,right_pred)
    


def main():
    # raw 
    imu_csv_path = "./rawdata/lyr2_ex.csv"
    time_csv_path = "./timestamp/ex_lyr2.csv"
    imu_data = pd.read_csv(imu_csv_path)
    imu_data = imu_data.values
    time_data = pd.read_csv(time_csv_path)
    time_data = time_data.values


    imu_data = imu_data[:,2:]
    imu_data[:,0:6] = unit_conversion(imu_data[:,0:6])
    imu_data[:,6:12] = unit_conversion(imu_data[:,6:12])
    imu_data[:,12:18] = unit_conversion(imu_data[:,12:18])
    mapping_uniform(imu_data)


    

if __name__ == "__main__":
    main()