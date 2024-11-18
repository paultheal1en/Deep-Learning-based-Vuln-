static int load_sofa(AVFilterContext *ctx, char *filename, int *samplingrate)

{

    struct SOFAlizerContext *s = ctx->priv;

    /* variables associated with content of SOFA file: */

    int ncid, n_dims, n_vars, n_gatts, n_unlim_dim_id, status;

    char data_delay_dim_name[NC_MAX_NAME];

    float *sp_a, *sp_e, *sp_r, *data_ir;

    char *sofa_conventions;

    char dim_name[NC_MAX_NAME];   /* names of netCDF dimensions */

    size_t *dim_length;           /* lengths of netCDF dimensions */

    char *text;

    unsigned int sample_rate;

    int data_delay_dim_id[2];

    int samplingrate_id;

    int data_delay_id;

    int n_samples;

    int m_dim_id = -1;

    int n_dim_id = -1;

    int data_ir_id;

    size_t att_len;

    int m_dim;

    int *data_delay;

    int sp_id;

    int i, ret;



    s->sofa.ncid = 0;

    status = nc_open(filename, NC_NOWRITE, &ncid); /* open SOFA file read-only */

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Can't find SOFA-file '%s'\n", filename);

        return AVERROR(EINVAL);

    }



    /* get number of dimensions, vars, global attributes and Id of unlimited dimensions: */

    nc_inq(ncid, &n_dims, &n_vars, &n_gatts, &n_unlim_dim_id);



    /* -- get number of measurements ("M") and length of one IR ("N") -- */

    dim_length = av_malloc_array(n_dims, sizeof(*dim_length));

    if (!dim_length) {

        nc_close(ncid);

        return AVERROR(ENOMEM);

    }



    for (i = 0; i < n_dims; i++) { /* go through all dimensions of file */

        nc_inq_dim(ncid, i, (char *)&dim_name, &dim_length[i]); /* get dimensions */

        if (!strncmp("M", (const char *)&dim_name, 1)) /* get ID of dimension "M" */

            m_dim_id = i;

        if (!strncmp("N", (const char *)&dim_name, 1)) /* get ID of dimension "N" */

            n_dim_id = i;

    }



    if ((m_dim_id == -1) || (n_dim_id == -1)) { /* dimension "M" or "N" couldn't be found */

        av_log(ctx, AV_LOG_ERROR, "Can't find required dimensions in SOFA file.\n");

        av_freep(&dim_length);

        nc_close(ncid);

        return AVERROR(EINVAL);

    }



    n_samples = dim_length[n_dim_id]; /* get length of one IR */

    m_dim     = dim_length[m_dim_id]; /* get number of measurements */



    av_freep(&dim_length);



    /* -- check file type -- */

    /* get length of attritube "Conventions" */

    status = nc_inq_attlen(ncid, NC_GLOBAL, "Conventions", &att_len);

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Can't get length of attribute \"Conventions\".\n");

        nc_close(ncid);

        return AVERROR_INVALIDDATA;

    }



    /* check whether file is SOFA file */

    text = av_malloc(att_len + 1);

    if (!text) {

        nc_close(ncid);

        return AVERROR(ENOMEM);

    }



    nc_get_att_text(ncid, NC_GLOBAL, "Conventions", text);

    *(text + att_len) = 0;

    if (strncmp("SOFA", text, 4)) {

        av_log(ctx, AV_LOG_ERROR, "Not a SOFA file!\n");

        av_freep(&text);

        nc_close(ncid);

        return AVERROR(EINVAL);

    }

    av_freep(&text);



    status = nc_inq_attlen(ncid, NC_GLOBAL, "License", &att_len);

    if (status == NC_NOERR) {

        text = av_malloc(att_len + 1);

        if (text) {

            nc_get_att_text(ncid, NC_GLOBAL, "License", text);

            *(text + att_len) = 0;

            av_log(ctx, AV_LOG_INFO, "SOFA file License: %s\n", text);

            av_freep(&text);

        }

    }



    status = nc_inq_attlen(ncid, NC_GLOBAL, "SourceDescription", &att_len);

    if (status == NC_NOERR) {

        text = av_malloc(att_len + 1);

        if (text) {

            nc_get_att_text(ncid, NC_GLOBAL, "SourceDescription", text);

            *(text + att_len) = 0;

            av_log(ctx, AV_LOG_INFO, "SOFA file SourceDescription: %s\n", text);

            av_freep(&text);

        }

    }



    status = nc_inq_attlen(ncid, NC_GLOBAL, "Comment", &att_len);

    if (status == NC_NOERR) {

        text = av_malloc(att_len + 1);

        if (text) {

            nc_get_att_text(ncid, NC_GLOBAL, "Comment", text);

            *(text + att_len) = 0;

            av_log(ctx, AV_LOG_INFO, "SOFA file Comment: %s\n", text);

            av_freep(&text);

        }

    }



    status = nc_inq_attlen(ncid, NC_GLOBAL, "SOFAConventions", &att_len);

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Can't get length of attribute \"SOFAConventions\".\n");

        nc_close(ncid);

        return AVERROR_INVALIDDATA;

    }



    sofa_conventions = av_malloc(att_len + 1);

    if (!sofa_conventions) {

        nc_close(ncid);

        return AVERROR(ENOMEM);

    }



    nc_get_att_text(ncid, NC_GLOBAL, "SOFAConventions", sofa_conventions);

    *(sofa_conventions + att_len) = 0;

    if (strncmp("SimpleFreeFieldHRIR", sofa_conventions, att_len)) {

        av_log(ctx, AV_LOG_ERROR, "Not a SimpleFreeFieldHRIR file!\n");

        av_freep(&sofa_conventions);

        nc_close(ncid);

        return AVERROR(EINVAL);

    }

    av_freep(&sofa_conventions);



    /* -- get sampling rate of HRTFs -- */

    /* read ID, then value */

    status  = nc_inq_varid(ncid, "Data.SamplingRate", &samplingrate_id);

    status += nc_get_var_uint(ncid, samplingrate_id, &sample_rate);

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Couldn't read Data.SamplingRate.\n");

        nc_close(ncid);

        return AVERROR(EINVAL);

    }

    *samplingrate = sample_rate; /* remember sampling rate */



    /* -- allocate memory for one value for each measurement position: -- */

    sp_a = s->sofa.sp_a = av_malloc_array(m_dim, sizeof(float));

    sp_e = s->sofa.sp_e = av_malloc_array(m_dim, sizeof(float));

    sp_r = s->sofa.sp_r = av_malloc_array(m_dim, sizeof(float));

    /* delay and IR values required for each ear and measurement position: */

    data_delay = s->sofa.data_delay = av_calloc(m_dim, 2 * sizeof(int));

    data_ir = s->sofa.data_ir = av_malloc_array(m_dim * n_samples, sizeof(float) * 2);



    if (!data_delay || !sp_a || !sp_e || !sp_r || !data_ir) {

        /* if memory could not be allocated */

        close_sofa(&s->sofa);

        return AVERROR(ENOMEM);

    }



    /* get impulse responses (HRTFs): */

    /* get corresponding ID */

    status = nc_inq_varid(ncid, "Data.IR", &data_ir_id);

    status += nc_get_var_float(ncid, data_ir_id, data_ir); /* read and store IRs */

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Couldn't read Data.IR!\n");

        ret = AVERROR(EINVAL);

        goto error;

    }



    /* get source positions of the HRTFs in the SOFA file: */

    status  = nc_inq_varid(ncid, "SourcePosition", &sp_id); /* get corresponding ID */

    status += nc_get_vara_float(ncid, sp_id, (size_t[2]){ 0, 0 } ,

                (size_t[2]){ m_dim, 1}, sp_a); /* read & store azimuth angles */

    status += nc_get_vara_float(ncid, sp_id, (size_t[2]){ 0, 1 } ,

                (size_t[2]){ m_dim, 1}, sp_e); /* read & store elevation angles */

    status += nc_get_vara_float(ncid, sp_id, (size_t[2]){ 0, 2 } ,

                (size_t[2]){ m_dim, 1}, sp_r); /* read & store radii */

    if (status != NC_NOERR) { /* if any source position variable coudn't be read */

        av_log(ctx, AV_LOG_ERROR, "Couldn't read SourcePosition.\n");

        ret = AVERROR(EINVAL);

        goto error;

    }



    /* read Data.Delay, check for errors and fit it to data_delay */

    status  = nc_inq_varid(ncid, "Data.Delay", &data_delay_id);

    status += nc_inq_vardimid(ncid, data_delay_id, &data_delay_dim_id[0]);

    status += nc_inq_dimname(ncid, data_delay_dim_id[0], data_delay_dim_name);

    if (status != NC_NOERR) {

        av_log(ctx, AV_LOG_ERROR, "Couldn't read Data.Delay.\n");

        ret = AVERROR(EINVAL);

        goto error;

    }



    /* Data.Delay dimension check */

    /* dimension of Data.Delay is [I R]: */

    if (!strncmp(data_delay_dim_name, "I", 2)) {

        /* check 2 characters to assure string is 0-terminated after "I" */

        int delay[2]; /* delays get from SOFA file: */



        av_log(ctx, AV_LOG_DEBUG, "Data.Delay has dimension [I R]\n");

        status = nc_get_var_int(ncid, data_delay_id, &delay[0]);

        if (status != NC_NOERR) {

            av_log(ctx, AV_LOG_ERROR, "Couldn't read Data.Delay\n");

            ret = AVERROR(EINVAL);

            goto error;

        }

        int *data_delay_r = data_delay + m_dim;

        for (i = 0; i < m_dim; i++) { /* extend given dimension [I R] to [M R] */

            /* assign constant delay value for all measurements to data_delay fields */

            data_delay[i]   = delay[0];

            data_delay_r[i] = delay[1];

        }

        /* dimension of Data.Delay is [M R] */

    } else if (!strncmp(data_delay_dim_name, "M", 2)) {

        av_log(ctx, AV_LOG_ERROR, "Data.Delay in dimension [M R]\n");

        /* get delays from SOFA file: */

        status = nc_get_var_int(ncid, data_delay_id, data_delay);

        if (status != NC_NOERR) {

            av_log(ctx, AV_LOG_ERROR, "Couldn't read Data.Delay\n");

            ret = AVERROR(EINVAL);

            goto error;

        }

    } else { /* dimension of Data.Delay is neither [I R] nor [M R] */

        av_log(ctx, AV_LOG_ERROR, "Data.Delay does not have the required dimensions [I R] or [M R].\n");

        ret = AVERROR(EINVAL);

        goto error;

    }



    /* save information in SOFA struct: */

    s->sofa.m_dim = m_dim; /* no. measurement positions */

    s->sofa.n_samples = n_samples; /* length on one IR */

    s->sofa.ncid = ncid; /* netCDF ID of SOFA file */

    nc_close(ncid); /* close SOFA file */



    return 0;



error:

    close_sofa(&s->sofa);

    return ret;

}
