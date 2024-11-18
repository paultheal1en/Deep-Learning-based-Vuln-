void register_module_init(void (*fn)(void), module_init_type type)

{

    ModuleEntry *e;

    ModuleTypeList *l;



    e = qemu_mallocz(sizeof(*e));

    e->init = fn;



    l = find_type(type);



    TAILQ_INSERT_TAIL(l, e, node);

}
