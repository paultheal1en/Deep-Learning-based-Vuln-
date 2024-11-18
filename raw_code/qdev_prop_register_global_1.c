void qdev_prop_register_global(GlobalProperty *prop)

{

    QTAILQ_INSERT_TAIL(&global_props, prop, next);

}
