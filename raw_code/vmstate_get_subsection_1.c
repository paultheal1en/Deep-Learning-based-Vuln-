    vmstate_get_subsection(const VMStateSubsection *sub, char *idstr)

{

    while (sub && sub->needed) {

        if (strcmp(idstr, sub->vmsd->name) == 0) {

            return sub->vmsd;

        }

        sub++;

    }

    return NULL;

}
