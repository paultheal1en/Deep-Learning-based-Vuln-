void error_propagate(Error **dst_errp, Error *local_err)

{

    if (local_err && dst_errp == &error_abort) {

        error_report_err(local_err);

        abort();

    } else if (dst_errp && !*dst_errp) {

        *dst_errp = local_err;

    } else if (local_err) {

        error_free(local_err);

    }

}
