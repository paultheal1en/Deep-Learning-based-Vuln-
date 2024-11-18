void error_propagate(Error **dst_err, Error *local_err)

{

    if (dst_err) {

        *dst_err = local_err;

    } else if (local_err) {

        error_free(local_err);

    }

}
