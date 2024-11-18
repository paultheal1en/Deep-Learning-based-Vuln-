static double setup_compress_thresh(double threshold)

{

    if ((threshold > DBL_EPSILON) && (threshold < (1.0 - DBL_EPSILON))) {

        double current_threshold = threshold;

        double step_size = 1.0;



        while (step_size > DBL_EPSILON) {

            while ((current_threshold + step_size > current_threshold) &&

                   (bound(current_threshold + step_size, 1.0) <= threshold)) {

                current_threshold += step_size;

            }



            step_size /= 2.0;

        }



        return current_threshold;

    } else {

        return threshold;

    }

}
