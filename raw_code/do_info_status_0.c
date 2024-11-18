static void do_info_status(Monitor *mon, QObject **ret_data)

{

    *ret_data = qobject_from_jsonf("{ 'running': %i, 'singlestep': %i }",

                                    runstate_is_running(), singlestep);

}
