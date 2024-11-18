static int test_scalarproduct_float(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                    const float *v1, const float *v2)

{

    float cprod, oprod;

    int ret;



    cprod = cdsp->scalarproduct_float(v1, v2, LEN);

    oprod = fdsp->scalarproduct_float(v1, v2, LEN);



    if (ret = compare_floats(&cprod, &oprod, 1, ARBITRARY_SCALARPRODUCT_CONST))

        av_log(NULL, AV_LOG_ERROR, "scalarproduct_float failed\n");



    return ret;

}
