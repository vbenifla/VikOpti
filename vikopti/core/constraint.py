class Constraint:
    """
    Class representing a constraint.
    """

    def __init__(self, type: str, limit: float):
        """
        Construct the constraint object and set the different attributes.

        Parameters
        ----------
        type : str
            constraint's type, "sup" for superior to or "inf" for inferior to.
        limit : float
            constraint's limit.
        """

        # set constraint's type
        self.type = type

        # set constraints' limit
        self.limit = limit
