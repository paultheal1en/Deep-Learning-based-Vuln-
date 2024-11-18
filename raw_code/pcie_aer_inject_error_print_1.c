void pcie_aer_inject_error_print(Monitor *mon, const QObject *data)

{

    QDict *qdict;

    int devfn;

    assert(qobject_type(data) == QTYPE_QDICT);

    qdict = qobject_to_qdict(data);



    devfn = (int)qdict_get_int(qdict, "devfn");

    monitor_printf(mon, "OK id: %s root bus: %s, bus: %x devfn: %x.%x\n",

                   qdict_get_str(qdict, "id"),

                   qdict_get_str(qdict, "root_bus"),

                   (int) qdict_get_int(qdict, "bus"),

                   PCI_SLOT(devfn), PCI_FUNC(devfn));

}
