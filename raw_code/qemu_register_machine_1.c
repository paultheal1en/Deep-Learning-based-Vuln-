int qemu_register_machine(QEMUMachine *m)

{

    TypeInfo ti = {

        .name       = g_strconcat(m->name, TYPE_MACHINE_SUFFIX, NULL),

        .parent     = TYPE_MACHINE,

        .class_init = machine_class_init,

        .class_data = (void *)m,

    };



    type_register(&ti);



    return 0;

}
