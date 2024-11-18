static int parse_iplconvkernel(IplConvKernel **kernel, char *buf, void *log_ctx)

{

    char shape_filename[128] = "", shape_str[32] = "rect";

    int cols = 0, rows = 0, anchor_x = 0, anchor_y = 0, shape = CV_SHAPE_RECT;

    int *values = NULL, ret;



    sscanf(buf, "%dx%d+%dx%d/%32[^=]=%127s", &cols, &rows, &anchor_x, &anchor_y, shape_str, shape_filename);



    if      (!strcmp(shape_str, "rect"   )) shape = CV_SHAPE_RECT;

    else if (!strcmp(shape_str, "cross"  )) shape = CV_SHAPE_CROSS;

    else if (!strcmp(shape_str, "ellipse")) shape = CV_SHAPE_ELLIPSE;

    else if (!strcmp(shape_str, "custom" )) {

        shape = CV_SHAPE_CUSTOM;

        if ((ret = read_shape_from_file(&cols, &rows, &values, shape_filename, log_ctx)) < 0)

            return ret;

    } else {

        av_log(log_ctx, AV_LOG_ERROR,

               "Shape unspecified or type '%s' unknown.\n", shape_str);

        return AVERROR(EINVAL);

    }



    if (rows <= 0 || cols <= 0) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Invalid non-positive values for shape size %dx%d\n", cols, rows);

        return AVERROR(EINVAL);

    }



    if (anchor_x < 0 || anchor_y < 0 || anchor_x >= cols || anchor_y >= rows) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Shape anchor %dx%d is not inside the rectangle with size %dx%d.\n",

               anchor_x, anchor_y, cols, rows);

        return AVERROR(EINVAL);

    }



    *kernel = cvCreateStructuringElementEx(cols, rows, anchor_x, anchor_y, shape, values);

    av_freep(&values);

    if (!*kernel)

        return AVERROR(ENOMEM);



    av_log(log_ctx, AV_LOG_VERBOSE, "Structuring element: w:%d h:%d x:%d y:%d shape:%s\n",

           rows, cols, anchor_x, anchor_y, shape_str);

    return 0;

}
