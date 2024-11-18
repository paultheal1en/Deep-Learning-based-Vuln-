static VirtIOSerialPort *find_port_by_name(char *name)

{

    VirtIOSerial *vser;



    QLIST_FOREACH(vser, &vserdevices.devices, next) {

        VirtIOSerialPort *port;



        QTAILQ_FOREACH(port, &vser->ports, next) {

            if (!strcmp(port->name, name)) {

                return port;

            }

        }

    }

    return NULL;

}
