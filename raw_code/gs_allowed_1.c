bool gs_allowed(void)

{

    /* for "none" machine this results in true */

    return get_machine_class()->gs_allowed;

}
