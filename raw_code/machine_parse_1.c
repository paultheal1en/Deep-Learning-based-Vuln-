static QEMUMachine *machine_parse(const char *name)

{

    QEMUMachine *m, *machine = NULL;



    if (name) {

        machine = find_machine(name);

    }

    if (machine) {

        return machine;

    }

    printf("Supported machines are:\n");

    for (m = first_machine; m != NULL; m = m->next) {

        if (m->alias) {

            printf("%-20s %s (alias of %s)\n", m->alias, m->desc, m->name);

        }

        printf("%-20s %s%s\n", m->name, m->desc,

               m->is_default ? " (default)" : "");

    }

    exit(!name || *name != '?');

}
