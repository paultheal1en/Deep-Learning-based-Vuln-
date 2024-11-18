static void qobject_output_type_uint64(Visitor *v, const char *name,

                                       uint64_t *obj, Error **errp)

{

    /* FIXME values larger than INT64_MAX become negative */

    QObjectOutputVisitor *qov = to_qov(v);

    qobject_output_add(qov, name, qnum_from_int(*obj));

}
