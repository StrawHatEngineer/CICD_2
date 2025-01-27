from .signature_verification import register as register_signature_verification


def register(registered_functions):
    '''
    Registers all functions allowed by ibtools.
    '''
    register_signature_verification(registered_functions)