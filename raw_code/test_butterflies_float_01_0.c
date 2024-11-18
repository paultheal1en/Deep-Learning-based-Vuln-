static int test_butterflies_float(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                  const float *v1, const float *v2)

{

    LOCAL_ALIGNED(32, float, cv1, [LEN]);

    LOCAL_ALIGNED(32, float, cv2, [LEN]);

    LOCAL_ALIGNED(32, float, ov1, [LEN]);

    LOCAL_ALIGNED(32, float, ov2, [LEN]);

    int ret;



    memcpy(cv1, v1, LEN * sizeof(*v1));

    memcpy(cv2, v2, LEN * sizeof(*v2));

    memcpy(ov1, v1, LEN * sizeof(*v1));

    memcpy(ov2, v2, LEN * sizeof(*v2));



    cdsp->butterflies_float(cv1, cv2, LEN);

    fdsp->butterflies_float(ov1, ov2, LEN);



    if ((ret = compare_floats(cv1, ov1, LEN, FLT_EPSILON)) ||

        (ret = compare_floats(cv2, ov2, LEN, FLT_EPSILON)))

        av_log(NULL, AV_LOG_ERROR, "butterflies_float failed\n");



    return ret;

}
