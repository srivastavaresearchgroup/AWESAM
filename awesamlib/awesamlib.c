#include<math.h>

double max(double * x, int start, int end) 
{
    double maximum = x[start];
    for (int i = start; i < end; i++) {
        if (x[i] > maximum)
            maximum = x[i];
    }
    return maximum;
}

int index_max(int a, int b) 
{
    if (a >= b)
        return a;
    else
        return b;
}

int index_min(int a, int b) 
{
    if (a <= b)
        return a;
    else
        return b;
}

void adaptive_maxfilter(double * x, int * kernels, double * out, int len_x, int len_kernels, int stride)
{
    int i = 0;
    while(i < len_kernels) {
        int lower_index = index_max(0, i * stride - kernels[i] / 2);
        int upper_index = index_min(len_x - 1, i * stride + kernels[i] / 2);

        out[i] = max(x, lower_index, upper_index);

        i++;
    }
}

int range(double * times, int min_index, int max_index, double t)
{
    int i;
    for (i = min_index; i < max_index; i++) {
        if (times[i] >= t) {
            return i;
        }
    }
    return i;
}

double compute_metric(double * metric, double dt, double dy, double amp)
{
    double x = metric[0]*dt + metric[1]*dy;
    double y = metric[2]*dt + metric[3]*dy;
    return sqrt(x*x +  y*y)/amp;
}

double min_double(double a, double b)
{
    if (a < b)
        return a;
    else
        return b;
}

double absolute(double a)
{
    if (a < 0)
        return -a;
    else
        return a;
}

double get_closest_event(
    double pc_time, double pc_amplitude,
    double * cc_times, double * cc_amplitudes,
    int index_min, int index_max, double * metric)
{
    double out_distance = -1;
    double dist;
    for (int i = index_min; i < index_max; i++) {
        dist = compute_metric(
            metric, 
            absolute(pc_time - cc_times[i]),
            absolute(pc_amplitude - cc_amplitudes[i]),
            min_double(pc_amplitude, cc_amplitudes[i])
        );

        if ((out_distance > dist) | (out_distance < 0)){
            out_distance = dist;
        }
    }
    return out_distance;
}



void compute_probabilities(
    double * pc_times, double * pc_amplitudes, 
    double * cc_times, double * cc_amplitudes, 
    double * out, double * metric, 
    int len_pc, int len_cc, double window_size) 
{
    // metric: ((a,b), (c,d)) ) where b,c = 0
    // catalog must be sorted

    double time, amplitude;
    int min_index = 0;
    int max_index = 0;
    double distance;
    
    for (int i = 0; i < len_pc; i++) {
        time = pc_times[i];
        amplitude = pc_amplitudes[i];

        min_index = range(cc_times, min_index, len_cc, time - window_size);
        max_index = range(cc_times, max_index, len_cc, time + window_size);

        if (min_index != max_index) {
            distance = get_closest_event(time, amplitude, cc_times, 
                cc_amplitudes, min_index, max_index, metric);
            out[i] = exp(-distance);
        } else {
            out[i] = 0; 
        }
    }
}


