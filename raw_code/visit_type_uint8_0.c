void visit_type_uint8(Visitor *v, uint8_t *obj, const char *name, Error **errp)

{

    int64_t value;



    if (v->type_uint8) {

        v->type_uint8(v, obj, name, errp);

    } else {

        value = *obj;

        v->type_int64(v, &value, name, errp);

        if (value < 0 || value > UINT8_MAX) {

            /* FIXME questionable reuse of errp if callback changed

               value on error */

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                       name ? name : "null", "uint8_t");

            return;

        }

        *obj = value;

    }

}
