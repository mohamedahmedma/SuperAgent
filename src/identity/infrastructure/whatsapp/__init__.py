"""WhatsApp: sending a code, reading a delivery, and knowing which school it is for."""
from identity.infrastructure.whatsapp.channels import (
    UNATTRIBUTABLE_SCHOOL,
    WhatsAppChannels,
    build_channels,
    build_registry,
)
from identity.infrastructure.whatsapp.gateways import (
    CloudApiWhatsAppGateway,
    RecordingWhatsAppGateway,
)
from identity.infrastructure.whatsapp.inbound import (
    InboundMessage,
    inbound_text_messages,
    signature_is_valid,
)

__all__ = [
    "UNATTRIBUTABLE_SCHOOL",
    "CloudApiWhatsAppGateway",
    "InboundMessage",
    "RecordingWhatsAppGateway",
    "WhatsAppChannels",
    "build_channels",
    "build_registry",
    "inbound_text_messages",
    "signature_is_valid",
]
