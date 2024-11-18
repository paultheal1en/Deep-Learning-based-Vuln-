static void qmp_input_stack_object_free(StackObject *tos)

{

    if (tos->h) {

        g_hash_table_unref(tos->h);

    }



    g_free(tos);

}
