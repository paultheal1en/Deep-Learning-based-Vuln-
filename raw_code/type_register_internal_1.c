static TypeImpl *type_register_internal(const TypeInfo *info)

{

    TypeImpl *ti = g_malloc0(sizeof(*ti));

    int i;



    g_assert(info->name != NULL);



    if (type_table_lookup(info->name) != NULL) {

        fprintf(stderr, "Registering `%s' which already exists\n", info->name);

        abort();

    }



    ti->name = g_strdup(info->name);

    ti->parent = g_strdup(info->parent);



    ti->class_size = info->class_size;

    ti->instance_size = info->instance_size;



    ti->class_init = info->class_init;

    ti->class_base_init = info->class_base_init;

    ti->class_finalize = info->class_finalize;

    ti->class_data = info->class_data;



    ti->instance_init = info->instance_init;

    ti->instance_post_init = info->instance_post_init;

    ti->instance_finalize = info->instance_finalize;



    ti->abstract = info->abstract;



    for (i = 0; info->interfaces && info->interfaces[i].type; i++) {

        ti->interfaces[i].typename = g_strdup(info->interfaces[i].type);

    }

    ti->num_interfaces = i;



    type_table_add(ti);



    return ti;

}
