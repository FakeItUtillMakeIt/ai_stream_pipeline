#include <iostream>
#include <vector>

using namespace std;

class KalmanFilter {
private:
    double xhat;        // estimated value
    double P;           // estimation error
    double xhatminus;   // predicted value
    double Pminus;      // prediction error
    double K;           // Kalman gain
    double R;           // measurement noise
    double Q;           // process noise

public:
    KalmanFilter(double process_variance, double measurement_variance) {
        xhat = 0.0;
        P = 1.0;
        xhatminus = 0.0;
        Pminus = 0.0;
        K = 0.0;
        R = measurement_variance;
        Q = process_variance;
    }

    double update(double measurement) {
        // Time update (prediction)
        xhatminus = xhat;
        Pminus = P + Q;

        // Measurement update (correction)
        K = Pminus / (Pminus + R);
        xhat = xhatminus + K * (measurement - xhatminus);
        P = (1 - K) * Pminus;

        return xhat;
    }
};

// int main() {
//     double process_variance = 1e-5;   // process noise
//     double measurement_variance = 0.1 * 0.1;  // measurement noise

//     KalmanFilter kf(process_variance, measurement_variance);

//     // Simulate real-time data stream
//     vector<double> data_stream = {10.1, 10.5, 9.8, 10.2, 10.0};  // example data stream

//     // Real-time updating
//     for (auto measurement : data_stream) {
//         double estimated_value = kf.update(measurement);
//         cout << "Current estimate: " << estimated_value << endl;
//     }

//     return 0;
// }
