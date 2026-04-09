{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

{% if modules %}
.. rubric:: Subpackages

.. autosummary::
   :toctree:

{% for item in modules %}
{% if (fullname + '.' + item) is package %}
   {{ fullname }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}

{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:

{% for item in modules %}
{% if not ((fullname + '.' + item) is package) %}
   {{ fullname }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}
