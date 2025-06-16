import {
  FileTextIcon,
  ChatBubbleIcon,
  InfoCircledIcon,
  ExclamationTriangleIcon,
} from "@radix-ui/react-icons";
import { BentoCard, BentoGrid } from "@/components/magicui/bento-grid";

const features = [
  {
    Icon: FileTextIcon,
    name: "Drug Information Dashboard",
    description:
      "Telusuri database lengkap obat-obatan: indikasi, efek samping, dosis, dan lainnya.",
    href: "/dashboard",
    cta: "Lihat dashboard",
    background: (
      <img
        src="/dashboard_preview.svg"
        alt="Dashboard illustration"
        className="absolute inset-0 w-full h-full object-cover opacity-20 z-0"
      />
    ),
  },
  {
    Icon: ChatBubbleIcon,
    name: "Drug Information Chatbot",
    description:
      "Tanyakan apa pun tentang obat dalam bahasa alami dan dapatkan jawaban akurat secara instan dari MediBot AI.",
    href: "/chatbot",
    cta: "Mulai percakapan",
    background: (
      <img
        src="/chatbot_preview.svg"
        alt="Chatbot illustration"
        className="absolute inset-0 w-full h-full object-cover opacity-20 z-0"
      />
    ),
  },
  {
    Icon: InfoCircledIcon,
    name: "Petunjuk Penggunaan",
    description:
      "• Ketik nama obat atau keluhan dengan jelas\n• Gunakan bahasa Indonesia atau Inggris\n• Periksa kembali ejaan untuk hasil yang akurat",
    href: "#",
    cta: "Pelajari lebih lanjut",
    background: (
      <div className="absolute inset-0 bg-blue-100 opacity-30 z-0" />
    ),
  },
  {
    Icon: ExclamationTriangleIcon,
    name: "Peringatan",
    description:
      "• MediBot bukan pengganti konsultasi dokter\n• Jangan lakukan diagnosis mandiri\n• Selalu konsultasikan dengan tenaga medis profesional",
    href: "#",
    cta: "Lihat detail",
    background: (
      <div className="absolute inset-0 bg-red-100 opacity-30 z-0" />
    ),
  },
];

export function BentoDemo() {
  return (
    <BentoGrid className="grid-cols-1 sm:grid-cols-2 gap-6 w-full">
      {features.map((feature) => (
        <BentoCard
          key={feature.name}
          {...feature}
          className="min-h-[280px] relative overflow-hidden"
        />
      ))}
    </BentoGrid>
  );
}
