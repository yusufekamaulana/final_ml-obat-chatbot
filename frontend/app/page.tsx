import { AuroraText } from "@/components/magicui/aurora-text";
import { BoxReveal } from "@/components/magicui/box-reveal";
import { RainbowButton } from "@/components/magicui/rainbow-button";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { BentoDemo } from "@/components/bento-section";

export default function HomePage() {
  return (
    <AuroraBackground>
      <div className="w-full min-h-screen overflow-y-auto px-4 py-16">
        <div className="transform scale-[0.75] origin-top mx-auto max-w-[1600px] flex flex-col gap-16">
          
          {/* SECTION: Selamat Datang - Full Width */}
          <div className="flex flex-col items-center text-center gap-6 w-full">
            <BoxReveal boxColor="#5046e6" duration={0.2}>
              <h1 className="text-4xl font-bold tracking-tight md:text-5xl lg:text-6xl">
                Selamat Datang di <AuroraText>MediBot</AuroraText>
              </h1>
            </BoxReveal>

            <BoxReveal boxColor="#5046e6" duration={0.2}>
              <h2 className="text-lg md:text-xl text-muted-foreground max-w-3xl">
                Asisten AI terpercaya untuk <AuroraText>informasi obat-obatan</AuroraText>
              </h2>
            </BoxReveal>

            <BoxReveal boxColor="#5046e6" duration={0.2}>
              <p className="text-sm md:text-base text-muted-foreground max-w-2xl">
                MediBot adalah partner cerdas kamu dalam mengakses informasi lengkap tentang obat — mulai dari panduan dosis, interaksi obat, efek samping, hingga indikasi medis. Baik kamu seorang tenaga kesehatan yang butuh referensi cepat, atau pasien yang ingin memahami resep dengan lebih jelas, MediBot siap membantu dengan akses instan dan percakapan interaktif. Ucapkan selamat tinggal pada kebingungan, dan sambut kepercayaan diri dalam pengambilan keputusan kesehatan kamu.
              </p>
            </BoxReveal>

            <BoxReveal boxColor="#5046e6" duration={0.6}>
              <div className="mt-2">
                <a href="/login">
                  <RainbowButton>Mulai Percakapan</RainbowButton>
                </a>
              </div>
            </BoxReveal>
          </div>

          {/* SECTION: Fitur (BentoGrid) */}
          <div className="w-full">
            <BentoDemo />
          </div>
        </div>
      </div>
    </AuroraBackground>
  );
}
