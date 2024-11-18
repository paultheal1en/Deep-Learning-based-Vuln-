void AUD_remove_card (QEMUSoundCard *card)

{

    LIST_REMOVE (card, entries);

    qemu_free (card->name);

}
