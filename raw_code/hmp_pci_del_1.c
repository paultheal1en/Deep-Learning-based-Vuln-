void hmp_pci_del(Monitor *mon, const QDict *qdict)

{

    pci_device_hot_remove(mon, qdict_get_str(qdict, "pci_addr"));

}
