void visit_type_uint8(Visitor *v, uint8_t *obj, const char *name, Error **errp)

{

    int64_t value;

    if (!error_is_set(errp)) {

        if (v->type_uint8) {

            v->type_uint8(v, obj, name, errp);

        } else {

            value = *obj;

            v->type_int(v, &value, name, errp);

            if (value < 0 || value > UINT8_MAX) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

                          "uint8_t");

                return;

            }

            *obj = value;

        }

    }

}
