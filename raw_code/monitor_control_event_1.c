static void monitor_control_event(void *opaque, int event)

{

    if (event == CHR_EVENT_OPENED) {

        QObject *data;

        Monitor *mon = opaque;



        mon->mc->command_mode = 0;

        json_message_parser_init(&mon->mc->parser, handle_qmp_command);



        data = get_qmp_greeting();

        monitor_json_emitter(mon, data);

        qobject_decref(data);

    }

}
